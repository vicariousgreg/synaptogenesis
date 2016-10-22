#include <vector>
#include "driver/stream.h"
#include "driver/scheduler.h"

void Stream::schedule_execution(int to_schedule) {
    while (scheduled < to_schedule) {
#ifdef PARALLEL
        Scheduler::get_instance()->schedule_execution(&this->cuda_stream, instructions[scheduled++]);
        for (int i = 0; i < IO_TYPE_SIZE; ++i)
            if (scheduled == last_index[i] + 1)
                cudaEventRecord(*events[i], cuda_stream);
#else
        Scheduler::get_instance()->schedule_execution(instructions[scheduled++]);
#endif
    }
}

void Stream::schedule_execution() {
    this->schedule_execution(instructions.size());
}

void Stream::schedule_execution(IOType type) {
    this->schedule_execution(last_index[type] + 1);
}

void Stream::schedule_weight_update() {
    for (int i = 0; i < instructions.size(); ++i)
#ifdef PARALLEL
        Scheduler::get_instance()->schedule_weight_update(&this->cuda_stream, instructions[i]);
#else
        Scheduler::get_instance()->schedule_weight_update(instructions[i]);
#endif
}

bool Stream::is_done() {
    return (scheduled == instructions.size()) and (not is_running());
}

bool Stream::is_done(IOType type) {
#ifdef PARALLEL
    return cudaEventQuery(*events[type]) == cudaSuccess;
#else
    return scheduled > last_index[type];
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
void Stream::wait_event(cudaEvent_t *event) {
    cudaStreamWaitEvent(this->cuda_stream, *event, 0);
}
#endif

void StreamCluster::reset() {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        it->second->reset();
}

void StreamCluster::schedule_execution() {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        it->second->schedule_execution();
}

void StreamCluster::schedule_execution(IOType type) {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        it->second->schedule_execution(type);
}

void StreamCluster::schedule_weight_update() {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        it->second->schedule_weight_update();
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
void StreamCluster::wait_event(cudaEvent_t *event) {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        it->second->wait_event(event);
}

void StreamCluster::block_stream(cudaStream_t stream) {
    for (auto it = streams.begin(); it != streams.end(); ++it)
        cudaStreamWaitEvent(stream, *it->second->finished_event, 0);
}

void StreamCluster::block_stream(cudaStream_t stream, IOType type) {
    for (auto it = streams.begin(); it != streams.end(); ++it) {
        for (int i = 0; i < IO_TYPE_SIZE; ++i)
            cudaStreamWaitEvent(stream, *it->second->events[i], 0);
    }
}
#endif
