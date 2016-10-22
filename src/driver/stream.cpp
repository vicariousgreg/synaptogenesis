#include <vector>
#include "driver/stream.h"
#include "driver/scheduler.h"

void Stream::execute(int to_execute) {
    while (executed < to_execute) {
#ifdef PARALLEL
        Scheduler::get_instance()->schedule_execution(&this->cuda_stream, instructions[executed++]);
        for (int i = 0; i < IO_TYPE_SIZE; ++i)
            if (executed == last_index[i] + 1)
                cudaEventRecord(events[i], cuda_stream);
#else
        Scheduler::get_instance()->schedule_execution(instructions[executed++]);
#endif
    }
}

void Stream::execute() {
    this->execute(instructions.size());
}

void Stream::execute(IOType type) {
    this->execute(last_index[type] + 1);
}

void Stream::update_weights() {
    for (int i = 0; i < instructions.size(); ++i)
#ifdef PARALLEL
        Scheduler::get_instance()->schedule_weight_update(&this->cuda_stream, instructions[i]);
#else
        Scheduler::get_instance()->schedule_weight_update(instructions[i]);
#endif
}

bool Stream::is_done() {
    return (executed == instructions.size()) and (not is_running());
}

bool Stream::is_done(IOType type) {
#ifdef PARALLEL
    return cudaEventQuery(events[type]) == cudaSuccess;
#else
    return executed > last_index[type];
#endif
}

bool Stream::is_running() {
#ifdef PARALLEL
    return cudaStreamQuery(this->cuda_stream) == cudaSuccess;
#else
    return false;
#endif
}

#ifdef PARALLEL
void Stream::wait_event(cudaEvent_t event) {
    cudaStreamWaitEvent(this->cuda_stream, event, 0);
}
#endif

void StreamCluster::reset() {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        it->second->reset();
}

void StreamCluster::execute() {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        it->second->execute();
}

void StreamCluster::execute(IOType type) {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        it->second->execute(type);
}

void StreamCluster::update_weights() {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        it->second->update_weights();
}

bool StreamCluster::is_done() {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        if (not it->second->is_done())
            return false;
    return true;
}

bool StreamCluster::is_done(IOType type) {
    for (auto it = streams.begin(); it != streams.end(); ++it) {
        if (not it->second->is_done(type))
            return false;
    }
    return true;
}

#ifdef PARALLEL
void StreamCluster::wait_event(cudaEvent_t event) {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        it->second->wait_event(event);
}

void StreamCluster::block_stream(cudaStream_t stream) {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        cudaStreamWaitEvent(stream, it->second->finished_event, 0);
}

void StreamCluster::block_stream(cudaStream_t stream, IOType type) {
    for (auto it = streams.begin(); it != streams.end(); ++it) {
        for (int i = 0; i < IO_TYPE_SIZE; ++i)
            cudaStreamWaitEvent(stream, it->second->events[i], 0);
    }
}
#endif
