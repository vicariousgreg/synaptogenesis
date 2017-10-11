#include "util/scheduler.h"

Scheduler *Scheduler::instance = 0;

Scheduler* Scheduler::get_instance() {
    if (Scheduler::instance == nullptr)
        Scheduler::instance = new Scheduler();
    return Scheduler::instance;
}

bool Scheduler::enqueued(Event *event) {
    std::unique_lock<std::mutex> lock(event_mutex);
    return enq.count(event);
}

void Scheduler::enqueue(Event *event) {
    std::unique_lock<std::mutex> e_lock(event_mutex);
    if (enq.count(event) > 0) {
        e_lock.unlock();
        std::unique_lock<std::mutex> lock(main_mutex);
        main_blocked_on = event;
        main_cv.wait(lock, [this, event](){return main_blocked_on == nullptr;});
        e_lock.lock();
    }
    enq.insert(event);
}

void Scheduler::dequeue(Event *event) {
    std::unique_lock<std::mutex> lock(event_mutex);
    enq.erase(event);
}

void Scheduler::push(Stream *stream, std::function<void()> f) {
    std::unique_lock<std::mutex> lock(stream_mutex[stream]);
    queues[stream].push(f);
}

void Scheduler::pop(Stream *stream) {
    std::unique_lock<std::mutex> lock(stream_mutex[stream]);
    queues[stream].pop();
}

bool Scheduler::freeze(Stream *stream, Event* event) {
    std::unique_lock<std::mutex> lock(event_mutex);
    if (enq.count(event)) {
        frozen_streams[event].push_back(stream);
        return true;
    }
    return false;
}

void Scheduler::thaw(Event* event) {
    std::unique_lock<std::mutex> lock(event_mutex);
    for (auto s : frozen_streams[event]) {
        mark_available(s);
    }
    frozen_streams[event].clear();
}

void Scheduler::mark_available(Stream *stream) {
    std::unique_lock<std::mutex> lock(stream_mutex[stream]);
    available_streams[stream] = true;
}

bool Scheduler::try_take(Stream *stream) {
    stream_mutex[stream].lock();
    if (available_streams[stream] and queues[stream].size() != 0) {
        available_streams[stream] = false;
        return true;
    }
    stream_mutex[stream].unlock();
    return false;
}

void Scheduler::release(Stream *stream) {
    stream_mutex[stream].unlock();
}

void Scheduler::notify_dormant() {
    std::unique_lock<std::mutex> lock(dormant_mutex);
    if (this->dormant) {
        this->dormant = false;
        dormant_cv.notify_all();
    }
}

void Scheduler::notify_main(Event *event) {
    std::unique_lock<std::mutex> lock(main_mutex);
    if (main_blocked_on == event) {
        main_blocked_on = nullptr;
        main_cv.notify_all();
    }
}


void Scheduler::add(Stream *stream) {
    streams.push_back(stream);
    stream_mutex[stream];
    available_streams[stream] = true;
    queues[stream] = std::queue<std::function<void()>>();
}

void Scheduler::enqueue_wait(Stream *stream, Event *event) {
    if (enqueued(event)) {
        notify_dormant();
        push(stream, std::bind(&Scheduler::wait, this, event, stream));
    }
}

void Scheduler::enqueue_record(Stream *stream, Event *event) {
    enqueue(event);
    notify_dormant();
    push(stream, std::bind(&Scheduler::record, this, event, stream));
}

void Scheduler::enqueue_compute(Stream *stream, std::function<void()> f) {
    notify_dormant();
    push(stream, std::bind(&Scheduler::compute, this, f, stream));
}

void Scheduler::synchronize(Event* event) {
    if (event->is_host()) {
        if (enqueued(event)) {
            std::unique_lock<std::mutex> lock(main_mutex);
            main_blocked_on = event;
            main_cv.wait(lock, [this](){return this->main_blocked_on == nullptr;});
            return;
        }
    }
#ifdef __CUDACC__
    else  {
        // Hosts wait on CUDA events by synchronizing
        cudaSetDevice(event->get_device_id());
        cudaEventSynchronize(event->cuda_event);
    }
#endif
}

void Scheduler::wait(Event* event, Stream* stream) {
    if (event->is_host()) {
        pop(stream);

        if (not freeze(stream, event))
            mark_available(stream);
    }
#ifdef __CUDACC__
    else if (stream->is_host()) {
        cudaSetDevice(event->get_device_id());
        if (cudaEventQuery(event->cuda_event)) {
            pop(stream);
        }
        mark_available(stream);
    } else {
        // Devices wait on CUDA events using CUDA API
        cudaSetDevice(event->get_device_id());
        cudaStreamWaitEvent(stream->cuda_stream, event->cuda_event, 0);

        pop(stream);
        mark_available(stream);
    }
#endif
}

void Scheduler::record(Event* event, Stream* stream) {
    pop(stream);
    dequeue(event);

    thaw(event);
    notify_main(event);
    mark_available(stream);
}

void Scheduler::compute(std::function<void()> f, Stream* stream) {
    pop(stream);
    f();
    mark_available(stream);
}

void Scheduler::start(int size) {
    if (running) shutdown();
    running = true;
    for (int i = 0 ; i < size ; ++i) {
        threads.push_back(std::thread(&Scheduler::worker_loop, this));
    }
}

void Scheduler::shutdown() {
    this->running = false;
    notify_dormant();

    for (auto& thread : threads)
        if (thread.joinable()) thread.join();

    threads.clear();
    for (auto pair : available_streams)
        pair.second = false;
    frozen_streams.clear();
    queues.clear();
}

void Scheduler::worker_loop() {
    while (running) {
        int size = streams.size();
        bool noop = true;

        for (int i = 0 ;  i < size; ++i) {
            auto stream = streams[i];
            if (try_take(stream)) {
                auto f = queues[stream].front();
                release(stream);
                noop = false;
                f();
                break;
            }
        }
        if (noop) {
            std::unique_lock<std::mutex> lock(dormant_mutex);
            this->dormant = true;
            dormant_cv.wait(lock, [this](){return not this->dormant;});
        }
    }
}
