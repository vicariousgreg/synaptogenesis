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
            // Flip the running flag and wake up any dormant threads
            std::unique_lock<std::mutex> lock(worker_mutex);
            this->pool_running = false;
            notify_dormant();
        }

        // Join the threads before cleaning up
        for (auto& thread : threads)
            if (thread.joinable()) thread.join();
    }

    // If verbose report on instruction completion by thread
    if (verbose and total_instructions != 0) {
        printf("Shutting down thread pool (total inst : %d).\n",
            total_instructions);

        int total_completed = 0;
        for (auto c : completed) {
            total_completed += c;
            printf("  Completed: %d\n", c);
        }
        printf("Total: %d\n\n", total_completed);
    }

    // Clean up
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
        // Synchronize with the event, blocking the caller if necessary
        synchronize(event);
    } else {
        {
            // The stream is considered waiting on the event if it has a wait
            //   instruction in its queue.  Keep track of the number of streams
            //   waiting for each event.  An event record cannot be reissued
            //   unless it is inactive and no streams are waiting for it
            std::unique_lock<std::mutex> lock(waiting_mutex);
            ++waiting_streams[event];
        }
        push(stream, std::bind(&Scheduler::wait, this, event, stream));
        notify_dormant();
    }
}

void Scheduler::enqueue_record(Stream *stream, Event *event) {
    if (single_thread) {
#ifdef __CUDACC__
        // Single threaded records for host events are no-ops
        // Device event records can be issued right away via CUDA API call
        if (not event->is_host())
            cudaEventRecord(event->cuda_event, stream->cuda_stream);
#endif
    } else {
        // Mark the event as active, meaning a record is enqueued
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
    // If necessary, block the caller until the event is recorded
    // Don't worry about streams waiting for the event
    maybe_block_client(event, false);
#ifdef __CUDACC__
    if (not event->is_host()) {
        // Once the record has been issued, the CUDA event must be waited on
        // Hosts wait on CUDA events by issuing a CUDA API synchronization
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

        // If available and non-empty queue, mark the stream unavailable, set
        //   the caller worker thread as owner, and return it
        std::unique_lock<std::mutex> lock(stream_mutex[stream]);
        if (available_streams[stream] and queues[stream].size() != 0) {
            assert(stream_owners[stream] == -1);
            available_streams[stream] = false;
            stream_owners[stream] = id;
            return stream;
        }
    }

    // If nothing was found, return nullptr
    // Caller will go dormant because there's nothing to do right now
    return nullptr;
}

void Scheduler::worker_run_stream(int id, Stream *stream) {
    // This is called when a worker has been assigned to a stream queue
    // Complete as many instruction as possible.  An instruction will return
    //   false if the queue must be halted (if the stream is frozen on an event)
    // If this happens, or if the queue is emptied, halt execution and return
    bool active = true;
    while (active) {
        // Get an instruction
        lock_stream(stream);
        auto f = queues[stream].front();
        unlock_stream(stream);

        // Execute the instruction
        if (f()) {
            // Instruction complete
            // Increment completed count and continue
            std::unique_lock<std::mutex> lock(stream_mutex[stream]);
            queues[stream].pop();
            ++completed[id];

            // Halt if the queue is empty
            if (queues[stream].size() == 0) {
                available_streams[stream] = true;
                stream_owners[stream] = -1;
                active = false;
            }
        } else active = false;
    }
}

void Scheduler::worker_loop(int id) {
    while (pool_running) {
        auto stream = worker_get_stream(id);

        if (stream != nullptr) {
            worker_run_stream(id, stream);
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
    bool ret = true;
    bool locked = false;
    bool release_stream = false;
    bool not_waiting = false;

    // Call maybe_freeze() to find out whether the event has been recorded
    // If true, the stream is frozen and the stream mutex is left locked
    // The stream will be thawed out once the event is recorded
    if (maybe_freeze(stream, event)) {
        locked = true;

        // Release the stream and return false
        release_stream = true;
        ret = false;

        // The caller will pop if the return value is true, but if the event is
        //   on the host, the instruction is complete (since the stream will be
        //   thawed and no CUDA API calls are necessary).  This is a
        //   special case where the queue needs to be popped even though
        //   false is returned, and the stream is not waiting anymore.
        // If the event is not a host event, CUDA API calls are still necessary
        //   once record is issued, so we cannot pop the instruction yet
        if (event->is_host()) {
            queues[stream].pop();
            ++completed[stream_owners[stream]];
            not_waiting = true;
        }
    } else {
        // If this is a host event, { maybe_freeze() == false } means that
        //   execution can continue, as no CUDA API calls are necessary
        if (event->is_host()) {
            not_waiting = true;
            ret = true;
        }
#ifdef __CUDACC__
        // CUDA event record has been issued, but may not have been completed
        else if (stream->is_host()) {
            // Host streams must issue CUDA queries to find out if the
            //   computation has been completed yet
            if (cudaEventQuery(event->cuda_event) == cudaSuccess) {
                // The event is completed, so the host can continue
                not_waiting = true;
                ret = true;
            } else {
                // The event has not been completed, so the host must wait
                // Don't pop the instruction; we need to come back here for
                //   another query later
                release_stream = true;
                ret = false;
            }
        } else {
            // If this stream is a CUDA stream, we can issue a CUDA wait and
            //   continue execution.  Subsequent instructions will wait until
            //   the event is completed on the device
            cudaStreamWaitEvent(stream->cuda_stream, event->cuda_event, 0);

            not_waiting = true;
            ret = true;
        }
#endif
    }

    // Lock the stream if maybe_freeze() didn't leave it locked
    if (not locked) lock_stream(stream);

    // Release the stream from this worker if necessary
    if (release_stream) stream_owners[stream] = -1;

    // Unlock the stream now
    unlock_stream(stream);

    // If the stream is no longer waiting for event record dispatch,
    //   decrement the wait count
    if (not_waiting) {
        std::unique_lock<std::mutex> lock(waiting_mutex);
        assert(waiting_streams[event] > 0);
        waiting_streams[event] -= 1;
    }
    return ret;
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
    ++total_instructions;
    std::unique_lock<std::mutex> lock(stream_mutex[stream]);
    queues[stream].push(f);
}

bool Scheduler::active(Event *event) {
    std::unique_lock<std::mutex> lock(event_mutex);
    return active_events[event];
}

void Scheduler::activate(Event *event) {
    // Events can only be issued if they are inactive and no streams are waiting
    //   on them (no streams have wait instructions in their queues)
    maybe_block_client(event, true);

    std::unique_lock<std::mutex> e_lock(event_mutex);
    active_events[event] = true;
}

void Scheduler::deactivate(Event *event) {
    // Thaw out any streams that are waiting on this event and no others
    int thawed = thaw(event);

    // Notify the client in case it's waiting on this event
    {
        std::unique_lock<std::mutex> lock(client_mutex);
        client_cv.notify_all();
    }

    // If any streams were thawed, notify any dormant workers
    if (thawed > 0) notify_dormant();
}

bool Scheduler::maybe_freeze(Stream *stream, Event* event) {
    // Freeze the stream if the event is active (record has been enqueued)
    lock_stream(stream);
    std::unique_lock<std::mutex> lock(event_mutex);
    if (active_events[event]) {
        // If the stream is active, keep the stream locked
        // This avoids race conditions
        // The caller must unlock the stream
        frozen_streams[event].push_back(stream);
        ++stream_frozen_on[stream];
        return true;
    } else {
        // If the event is inactive, unlock the stream
        unlock_stream(stream);
        return false;
    }
}

int Scheduler::thaw(Event *event) {
    // Thaw out any streams if this is the last event they are waiting on
    int thawed = 0;

    std::unique_lock<std::mutex> lock(event_mutex);
    active_events[event] = false;
    for (auto stream : frozen_streams[event]) {
        // Only thaw the stream if this is the last event it's waiting on
        std::unique_lock<std::mutex> lock(stream_mutex[stream]);
        if ((stream_frozen_on[stream] -= 1) == 0) {
            available_streams[stream] = true;
            ++thawed;
        }
    }
    frozen_streams[event].clear();
    return thawed;
}


void Scheduler::maybe_block_client(Event *event, bool wait_on_streams) {
    std::unique_lock<std::mutex> e_lock(event_mutex);
    std::unique_lock<std::mutex> lock(client_mutex);
    {
        // If the event is inactive and we either don't care about waiting
        //   streams or there are none of them, don't block the caller
        std::unique_lock<std::mutex> lock(waiting_mutex);
        if (not active_events[event] and
            (not wait_on_streams or waiting_streams[event] == 0)) return;
    }

    notify_dormant();
    e_lock.unlock();

    // Block the caller until the above conditions are satisfied
    client_cv.wait(lock, [this, wait_on_streams, event]()
        {return not active_events[event] and
         (not wait_on_streams or waiting_streams[event] == 0);});
}

void Scheduler::notify_dormant() {
    // If any threads are dormant, wake them up
    std::unique_lock<std::mutex> lock(dormant_mutex);
    if (this->dormant) {
        this->dormant = false;
        dormant_cv.notify_all();
    }
}


void Scheduler::add(Stream *stream) {
    // Adds a stream to the scheduler
    std::unique_lock<std::mutex> lock(worker_mutex);
    if (std::find(streams.begin(), streams.end(), stream) == streams.end()) {
        lock_stream(stream);

        // Set up resources
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
    // Removes a stream from the scheduler
    std::unique_lock<std::mutex> lock(worker_mutex);
    auto it = std::find(streams.begin(), streams.end(), stream);
    if (it != streams.end()) {
        lock_stream(stream);

        // Clean up resources
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
    // Adds an event to the scheduler
    std::unique_lock<std::mutex> e_lock(event_mutex);
    std::unique_lock<std::mutex> lock(worker_mutex);

    // Set up resources
    events.insert(event);
    active_events[event] = false;
    waiting_streams[event] = 0;
    frozen_streams[event];
}

void Scheduler::remove(Event *event) {
    // Removes an event from the scheduler
    std::unique_lock<std::mutex> e_lock(event_mutex);
    std::unique_lock<std::mutex> lock(worker_mutex);

    // Clean up resources
    events.erase(event);
    active_events.erase(event);
    waiting_streams.erase(event);
    frozen_streams.erase(event);
}
