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
    for (auto s : frozen_streams[event]) {
        mark_available(s);
    }
    frozen_streams[event].clear();
    notify_main(event);
}

void Scheduler::push(Stream *stream, std::function<Scheduler::QueueSignal()> f) {
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
        return false;
    }
    return true;
}

void Scheduler::mark_available(Stream *stream) {
    std::unique_lock<std::mutex> lock(stream_mutex[stream]);
    available_streams[stream] = true;
}

bool Scheduler::try_take(Stream *stream) {
    std::unique_lock<std::mutex> lock(stream_mutex[stream]);
    if (available_streams[stream] and queues[stream].size() != 0) {
        available_streams[stream] = false;
        return true;
    }
    return false;
}

void Scheduler::lock(Stream *stream) {
    stream_mutex[stream].lock();
}

void Scheduler::unlock(Stream *stream) {
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
    queues[stream];
}

void Scheduler::enqueue_wait(Stream *stream, Event *event) {
    if (single_thread) {
        synchronize(event);
    } else {
        if (enqueued(event)) {
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
        enqueue(event);
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
    dequeue(event);
    return CONTINUE;
}

Scheduler::QueueSignal Scheduler::compute(std::function<void()> f, Stream* stream) {
    f();
    return CONTINUE;
}

void Scheduler::start(int size) {
    if (running) shutdown();
    if (size == 0) {
        single_thread = true;
    } else {
        running = true;
        for (int i = 0 ; i < size ; ++i) {
            threads.push_back(std::thread(&Scheduler::worker_loop, this));
        }
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

        worker_mutex.lock();
        int i = 0;
        for ( ; i < size; ++i) {
            auto stream = streams[index];
            index = (index + 1) % size;

            if (try_take(stream)) {
                worker_mutex.unlock();

                bool active = true;
                while (active) {
                    lock(stream);
                    if (queues[stream].size() == 0) {
                        available_streams[stream] = true;
                        unlock(stream);
                        active = false;
                    } else {
                        auto f = queues[stream].front();
                        unlock(stream);

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
                    }
                }
                break;
            }
        }

        // Couldn't get a stream -- go dormant
        if (i == size) {
            worker_mutex.unlock();
            std::unique_lock<std::mutex> lock(dormant_mutex);

            // If not running, a shutdown has been issued
            // This is necessary to avoid race conditions
            if (running) {
                this->dormant = true;
                dormant_cv.wait(lock, [this](){return not this->dormant;});
            }
        }
    }
}
