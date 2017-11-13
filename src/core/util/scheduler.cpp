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
    shutdown_thread_pool();

    total_instructions = 0;
    if (size == 0) {
        pool_running = false;
        single_thread = true;
    } else {
        pool_running = true;
        single_thread = false;
        for (int i = 0 ; i < size ; ++i) {
            threads.push_back(
                std::thread(&Scheduler::worker_loop, this, threads.size()));
            completed.push_back(0);
        }
    }
}

void Scheduler::shutdown_thread_pool(bool verbose) {
    if (this->pool_running) {
        {
            std::unique_lock<std::mutex> lock(worker_mutex);
            this->pool_running = false;
            notify_dormant();
        }

        for (auto& thread : threads)
            if (thread.joinable()) thread.join();
    }

    if (verbose and total_instructions != 0) {
        printf("Shutting down thread pool (total inst : %d).\n", total_instructions);
        int total_completed = 0;
        for (auto c : completed) {
            total_completed += c;
            printf("  Completed: %d\n", c);
        }
        printf("Total: %d\n\n", total_completed);
    }

    threads.clear();
    completed.clear();
    for (auto pair : available_streams)
        pair.second = false;
    frozen_streams.clear();
    queues.clear();
    active_events.clear();
    waiting_streams.clear();
    for (auto pair : stream_owners)
        pair.second = -1;
    for (auto pair : stream_frozen_on)
        pair.second = 0;
}

void Scheduler::enqueue_wait(Stream *stream, Event *event) {
    if (single_thread) {
        synchronize(event);
    } else {
        {
            std::unique_lock<std::mutex> lock(waiting_mutex);
            waiting_streams[event] += 1;
        }
        push(stream, std::bind(&Scheduler::wait, this, event, stream));
        notify_dormant();
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
    maybe_block_client(event, false);
#ifdef __CUDACC__
    if (not event->is_host()) {
        // Hosts wait on CUDA events by synchronizing
        cudaEventSynchronize(event->cuda_event);
    }
#endif
}


/******************************************************************************/
/*************************** WORKER FUNCTIONS *********************************/
/******************************************************************************/
Stream* Scheduler::worker_get_stream(int id) {
    std::unique_lock<std::mutex> lock(worker_mutex);

    // Do one pass over streams, trying each one
    int size = streams.size();
    for (int i = 0 ; i < size; ++i) {
        auto stream = streams[index];
        index = (index + 1) % size;

        // If available and non-empty queue, set unavailable and return
        std::unique_lock<std::mutex> lock(stream_mutex[stream]);
        if (available_streams[stream] and queues[stream].size() != 0) {
            assert(stream_owners[stream] == -1);
            available_streams[stream] = false;
            stream_owners[stream] = id;
            return stream;
        }
    }

    // If nothing was found, return nullptr
    return nullptr;
}

void Scheduler::worker_run_stream(Stream *stream, int id) {
    bool active = true;
    while (active) {
        lock_stream(stream);
        auto f = queues[stream].front();
        unlock_stream(stream);

        if (f()) {
            std::unique_lock<std::mutex> lock(stream_mutex[stream]);
            queues[stream].pop();
            completed[id] += 1;
        } else active = false;

        lock_stream(stream);
        if (active and queues[stream].size() == 0) {
            available_streams[stream] = true;
            stream_owners[stream] = -1;
            active = false;
        }
        unlock_stream(stream);
    }
}

void Scheduler::worker_loop(int id) {
    while (pool_running) {
        auto stream = worker_get_stream(id);

        if (stream != nullptr) {
            worker_run_stream(stream, id);
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

bool Scheduler::wait(Event* event, Stream* stream) {
    if (event->is_host()) {
        if (not freeze(stream, event)) {
            completed[stream_owners[stream]] += 1;
            queues[stream].pop();
            stream_owners[stream] = -1;
            unlock_stream(stream);
            std::unique_lock<std::mutex> lock(waiting_mutex);
            assert(waiting_streams[event] > 0);
            waiting_streams[event] -= 1;
            return false;
        } else {
            std::unique_lock<std::mutex> lock(waiting_mutex);
            assert(waiting_streams[event] > 0);
            waiting_streams[event] -= 1;
            return true;
        }
    }
#ifdef __CUDACC__
    else if (stream->is_host()) {
        if (freeze(stream, event)) {
            if (cudaEventQuery(event->cuda_event) == cudaSuccess) {
                std::unique_lock<std::mutex> lock(waiting_mutex);
                assert(waiting_streams[event] > 0);
                waiting_streams[event] -= 1;
                return true;
            } else {
                stream_owners[stream] = -1;
                unlock_stream(stream);
                return false;
            }
        } else return false;
    } else {
        if (freeze(stream, event)) {
            // Devices wait on CUDA events using CUDA API
            cudaStreamWaitEvent(stream->cuda_stream, event->cuda_event, 0);

            std::unique_lock<std::mutex> lock(waiting_mutex);
            assert(waiting_streams[event] > 0);
            waiting_streams[event] -= 1;
            return true;
        } else {
            stream_owners[stream] = -1;
            unlock_stream(stream);
            return false;
        }
    }
#endif
}

bool Scheduler::record(Event* event, Stream* stream) {
#ifdef __CUDACC__
    if (not event->is_host())
        cudaEventRecord(event->cuda_event, stream->cuda_stream);
#endif
    deactivate(event);
    return true;
}

bool Scheduler::compute(std::function<void()> f, Stream* stream) {
    f();
    return true;
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

void Scheduler::push(Stream *stream, std::function<bool()> f) {
    total_instructions += 1;
    std::unique_lock<std::mutex> lock(stream_mutex[stream]);
    queues[stream].push(f);
}

bool Scheduler::active(Event *event) {
    std::unique_lock<std::mutex> lock(event_mutex);
    return active_events[event];
}

void Scheduler::activate(Event *event) {
    maybe_block_client(event, true);

    std::unique_lock<std::mutex> e_lock(event_mutex);
    active_events[event] = true;
}

void Scheduler::deactivate(Event *event) {
    thaw(event);
    notify_client(event);
}

bool Scheduler::freeze(Stream *stream, Event* event) {
    lock_stream(stream);
    std::unique_lock<std::mutex> lock(event_mutex);
    if (active_events[event]) {
        frozen_streams[event].push_back(stream);
        stream_frozen_on[stream] += 1;
        return false;
    }
    unlock_stream(stream);
    return true;
}

void Scheduler::thaw(Event *event) {
    std::unique_lock<std::mutex> lock(event_mutex);
    active_events[event] = false;
    for (auto stream : frozen_streams[event]) {
        std::unique_lock<std::mutex> lock(stream_mutex[stream]);
        if ((stream_frozen_on[stream] -= 1) == 0)
            available_streams[stream] = true;
    }
    frozen_streams[event].clear();
}


void Scheduler::maybe_block_client(Event *event, bool wait_on_streams) {
    std::unique_lock<std::mutex> e_lock(event_mutex);
    std::unique_lock<std::mutex> lock(client_mutex);
    {
        std::unique_lock<std::mutex> lock(waiting_mutex);
        if (not active_events[event] and
            (not wait_on_streams or waiting_streams[event] == 0)) return;
    }

    notify_dormant();
    e_lock.unlock();

    client_cv.wait(lock, [this, wait_on_streams, event]()
        {return not active_events[event] and
         (not wait_on_streams or waiting_streams[event] == 0);});
}

void Scheduler::notify_client(Event *event) {
    std::unique_lock<std::mutex> lock(client_mutex);
    client_cv.notify_all();
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
        stream_owners[stream] = -1;
        stream_frozen_on[stream] = 0;
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
        stream_owners.erase(stream);
        stream_frozen_on.erase(stream);
        queues.erase(stream);
        unlock_stream(stream);
        index = 0;
    }
}

void Scheduler::add(Event *event) {
    std::unique_lock<std::mutex> e_lock(event_mutex);
    std::unique_lock<std::mutex> lock(worker_mutex);

    events.insert(event);
    active_events[event] = false;
    waiting_streams[event] = 0;
    frozen_streams[event];
}

void Scheduler::remove(Event *event) {
    std::unique_lock<std::mutex> e_lock(event_mutex);
    std::unique_lock<std::mutex> lock(worker_mutex);

    events.erase(event);
    active_events.erase(event);
    waiting_streams.erase(event);
    frozen_streams.erase(event);
}
