#include <algorithm>

#include "util/scheduler.h"

Scheduler *Scheduler::instance = 0;

Scheduler* Scheduler::get_instance() {
    if (Scheduler::instance == nullptr)
        Scheduler::instance = new Scheduler();
    return Scheduler::instance;
}

Scheduler::~Scheduler() {
    shutdown_thread_pool();
    Scheduler::instance = nullptr;
}

/******************************************************************************/
/*************************** CLIENT INTERFACE *********************************/
/******************************************************************************/
void Scheduler::start_thread_pool(unsigned int size) {
    if (pool_running) shutdown_thread_pool();

    if (size == 0) {
        pool_running = false;
        single_thread = true;
    } else {
        pool_running = true;
        single_thread = false;
        for (int i = 0 ; i < size ; ++i)
            threads.push_back(
                std::thread(&Scheduler::worker_loop, this));
    }
}

void Scheduler::shutdown_thread_pool() {
    if (this->pool_running) {
        {
            std::unique_lock<std::mutex> lock(worker_mutex);
            this->pool_running = false;
            notify_dormant();
        }

        for (auto& thread : threads)
            if (thread.joinable()) thread.join();
    }

    threads.clear();
    for (auto pair : available_streams)
        pair.second = false;
    frozen_streams.clear();
    queues.clear();
    active_events.clear();
}

void Scheduler::enqueue_wait(Stream *stream, Event *event) {
    if (single_thread) {
        synchronize(event);
    } else {
        if (active(event)) {
            push(stream, std::bind(&Scheduler::wait, this, event, stream));
            notify_dormant();
        }
    }
}

void Scheduler::enqueue_record(Stream *stream, Event *event) {
    if (single_thread) {
#ifdef __CUDACC__
        if (not event->is_host()) {
            cudaEventRecord(event->cuda_event, stream->cuda_stream);
        }
#endif
    } else {
        activate(event);
        push(stream, std::bind(&Scheduler::record, this, event, stream));
        notify_dormant();
    }
}

void Scheduler::enqueue_compute(Stream *stream, std::function<void()> f) {
    if (single_thread) {
        f();
    } else {
        push(stream, std::bind(&Scheduler::compute, this, f, stream));
        notify_dormant();
    }
}

void Scheduler::synchronize(Event* event) {
    if (event->is_host()) {
        maybe_block_client(event);
    }
#ifdef __CUDACC__
    else  {
        // Hosts wait on CUDA events by synchronizing
        cudaSetDevice(event->get_device_id());
        cudaEventSynchronize(event->cuda_event);
    }
#endif
}


/******************************************************************************/
/*************************** WORKER FUNCTIONS *********************************/
/******************************************************************************/
Stream* Scheduler::worker_get_stream() {
    std::unique_lock<std::mutex> lock(worker_mutex);

    // Do one pass over streams, trying each one
    int size = streams.size();
    for (int i = 0 ; i < size; ++i) {
        auto stream = streams[index];
        index = (index + 1) % size;

        // If available and non-empty queue, set unavailable and return
        std::unique_lock<std::mutex> lock(stream_mutex[stream]);
        if (available_streams[stream] and queues[stream].size() != 0) {
            available_streams[stream] = false;
            return stream;
        }
    }

    // If nothing was found, return nullptr
    return nullptr;
}

void Scheduler::worker_run_stream(Stream *stream) {
    bool active = true;
    while (active) {
        lock_stream(stream);
        auto f = queues[stream].front();
        unlock_stream(stream);

        switch(f()) {
            case STOP_POP:
                active = false;
            case CONTINUE:
                pop(stream);
                break;
            case STOP_NO_POP:
                active = false;
                mark_available(stream);
                break;
        }

        lock_stream(stream);
        if (queues[stream].size() == 0) {
            available_streams[stream] = true;
            active = false;
        }
        unlock_stream(stream);
    }
}

void Scheduler::worker_loop() {
    while (pool_running) {
        auto stream = worker_get_stream();

        if (stream != nullptr) {
            worker_run_stream(stream);
        } else {
            // Couldn't get a stream -- go dormant
            std::unique_lock<std::mutex> lock(dormant_mutex);

            // If not running, a shutdown has been issued
            // This is necessary to avoid race conditions
            if (pool_running) {
                this->dormant = true;
                dormant_cv.wait(lock, [this](){return not this->dormant;});
            }
        }
    }
}

Scheduler::QueueSignal Scheduler::wait(Event* event, Stream* stream) {
    if (event->is_host()) {
        return (freeze(stream, event)) ? CONTINUE : STOP_POP;
    }
#ifdef __CUDACC__
    else if (stream->is_host()) {
        cudaSetDevice(event->get_device_id());
        return
            (cudaEventQuery(event->cuda_event) == cudaSuccess)
                ? CONTINUE
                : STOP_NO_POP;
    } else {
        // Devices wait on CUDA events using CUDA API
        cudaSetDevice(event->get_device_id());
        cudaStreamWaitEvent(stream->cuda_stream, event->cuda_event, 0);

        return CONTINUE;
    }
#endif
}

Scheduler::QueueSignal Scheduler::record(Event* event, Stream* stream) {
#ifdef __CUDACC__
    if (not event->is_host())
        cudaEventRecord(event->cuda_event, stream->cuda_stream);
#endif
    deactivate(event);
    return CONTINUE;
}

Scheduler::QueueSignal Scheduler::compute(std::function<void()> f, Stream* stream) {
    f();
    return CONTINUE;
}


/******************************************************************************/
/*************************** UTILITY FUNCTIONS ********************************/
/******************************************************************************/
void Scheduler::lock_stream(Stream *stream) {
    stream_mutex[stream].lock();
}

void Scheduler::unlock_stream(Stream *stream) {
    stream_mutex[stream].unlock();
}

void Scheduler::push(Stream *stream, std::function<Scheduler::QueueSignal()> f) {
    std::unique_lock<std::mutex> lock(stream_mutex[stream]);
    queues[stream].push(f);
}

void Scheduler::pop(Stream *stream) {
    std::unique_lock<std::mutex> lock(stream_mutex[stream]);
    queues[stream].pop();
}

void Scheduler::mark_available(Stream *stream) {
    std::unique_lock<std::mutex> lock(stream_mutex[stream]);
    available_streams[stream] = true;
}


bool Scheduler::active(Event *event) {
    std::unique_lock<std::mutex> lock(event_mutex);
    return active_events[event];
}

void Scheduler::activate(Event *event) {
    maybe_block_client(event);

    std::unique_lock<std::mutex> e_lock(event_mutex);
    active_events[event] = true;
}

void Scheduler::deactivate(Event *event) {
    thaw(event);
    notify_client(event);
}

bool Scheduler::freeze(Stream *stream, Event* event) {
    std::unique_lock<std::mutex> lock(event_mutex);
    if (active_events[event]) {
        frozen_streams[event].push_back(stream);
        return false;
    }
    return true;
}

void Scheduler::thaw(Event *event) {
    std::unique_lock<std::mutex> lock(event_mutex);
    active_events[event] = false;
    for (auto s : frozen_streams[event]) {
        mark_available(s);
    }
    frozen_streams[event].clear();
}


void Scheduler::maybe_block_client(Event *event) {
    std::unique_lock<std::mutex> e_lock(event_mutex);
    if (not active_events[event]) return;

    notify_dormant();

    std::unique_lock<std::mutex> lock(client_mutex);

    client_blocked_on = event;
    e_lock.unlock();

    client_cv.wait(lock, [this, event]()
        {return client_blocked_on == nullptr;});
}

void Scheduler::notify_client(Event *event) {
    std::unique_lock<std::mutex> lock(client_mutex);
    if (client_blocked_on == event) {
        client_blocked_on = nullptr;
        client_cv.notify_all();
    }
}


void Scheduler::notify_dormant() {
    std::unique_lock<std::mutex> lock(dormant_mutex);
    if (this->dormant) {
        this->dormant = false;
        dormant_cv.notify_all();
    }
}


void Scheduler::add(Stream *stream) {
    std::unique_lock<std::mutex> lock(worker_mutex);
    if (std::find(streams.begin(), streams.end(), stream) == streams.end()) {
        lock_stream(stream);

        streams.push_back(stream);
        stream_mutex[stream];
        available_streams[stream] = true;
        queues[stream];

        unlock_stream(stream);
    }
}

void Scheduler::remove(Stream *stream) {
    std::unique_lock<std::mutex> lock(worker_mutex);
    auto it = std::find(streams.begin(), streams.end(), stream);
    if (it != streams.end()) {
        lock_stream(stream);

        streams.erase(it);
        stream_mutex.erase(stream);
        available_streams.erase(stream);
        queues.erase(stream);
        index = 0;

        unlock_stream(stream);
    }
}

void Scheduler::add(Event *event) {
    std::unique_lock<std::mutex> e_lock(event_mutex);
    std::unique_lock<std::mutex> lock(worker_mutex);

    events.insert(event);
    active_events[event] = false;
    frozen_streams[event];
}

void Scheduler::remove(Event *event) {
    std::unique_lock<std::mutex> e_lock(event_mutex);
    std::unique_lock<std::mutex> lock(worker_mutex);

    events.erase(event);
    active_events.erase(event);
    frozen_streams.erase(event);
}
