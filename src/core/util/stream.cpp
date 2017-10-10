#include "util/stream.h"
#include "util/event.h"
#include "util/pointer.h"

Stream::Stream(DeviceID device_id, bool host_flag)
        : device_id(device_id), host_flag(host_flag) {
    // Host streams launch worker threads
    if (host_flag) {
        this->running = true;
        this->waiting = true;
        this->thread = std::thread(&Stream::worker_loop, this);
    }
#ifdef __CUDACC__
    // Device streams use CUDA streams
    else {
        cudaSetDevice(device_id);
        cudaStreamCreate(&this->cuda_stream);
    }
#endif
}

Stream::~Stream() {
    if (host_flag) {
        // Flush the queue
        this->flush();

        // Synchronize so the worker is waiting
        this->synchronize();

        // Set running to false and wake up the thread
        this->running = false;
        this->waiting = false;
        cv.notify_one();

        // Join
        if (thread.joinable()) thread.join();
    }
#ifdef __CUDACC__
    else if (cuda_stream != 0) {
        cudaSetDevice(device_id);
        cudaStreamDestroy(this->cuda_stream);
    }
#endif
}

void Stream::schedule(std::function<void()> f) {
    if (host_flag) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(f);

        // Wake up thread if it's waiting
        if (this->waiting) {
            this->waiting = false;
            cv.notify_one();
        }
    } else f();
}

void Stream::record(Event *event) {
    event->mark_enqueued(this);
    if (host_flag)
        this->schedule(std::bind(&Event::mark_done, event));
}

void Stream::wait(Event *event) {
    if (event->is_enqueued())
        this->schedule(std::bind(&Event::wait, event, this));
}

void Stream::flush() {
    if (host_flag) {
        std::lock_guard<std::mutex> lock(mutex);

        // Clear the queue
        while (not queue.empty()) queue.pop();
    }
}

void Stream::synchronize() {
    // Wait for queue
    if (host_flag) while (not this->waiting);
}

void Stream::worker_loop() {
    while (running) {
        std::unique_lock<std::mutex> lock(mutex);

        // Wait for queue
        if (this->waiting)
            cv.wait(lock, [this](){return not this->waiting;});

        // Process the queue
        while (not queue.empty()) {
            auto f = queue.front();
            queue.pop();
            lock.unlock();
            try {
                f();
            } catch (...) {
                running = false;
                break;
            }
            lock.lock();
        }
        this->waiting = true;
    }
}

DefaultStream::DefaultStream(DeviceID device_id, bool host_flag) {
    this->device_id = device_id;
    this->host_flag = host_flag;
    this->running = false;
#ifdef __CUDACC__
    this->cuda_stream = 0;
#endif
}

void DefaultStream::schedule(std::function<void()> f) {
    f();
}

void DefaultStream::record(Event *event) {
    event->mark_enqueued(this);
    event->mark_done();
}

void DefaultStream::wait(Event *event) {
}
