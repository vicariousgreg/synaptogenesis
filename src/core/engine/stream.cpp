#include "engine/stream.h"

Stream::Stream(Layer *layer) : to_layer(layer), scheduled(0) {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        last_index[i] = 0;
#ifdef PARALLEL
    cudaStreamCreate(&cuda_stream);
    finished_event = new cudaEvent_t;
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        events[i] = new cudaEvent_t;
#endif
    reset();
}

Stream::~Stream() {
    for (int i = 0; i < this->instructions.size(); ++i)
        delete this->instructions[i];
#ifdef PARALLEL
    for (int i = 0; i < IO_TYPE_SIZE; ++i) {
        cudaEventDestroy(*events[i]);
        delete events[i];
    }
    cudaEventDestroy(*finished_event);
    delete finished_event;
#endif
}

void Stream::reset() {
    this->scheduled = 0;
#ifdef PARALLEL
    cudaEventCreateWithFlags(finished_event, cudaEventDisableTiming);
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        cudaEventCreateWithFlags(events[i], cudaEventDisableTiming);
#endif
}

void Stream::schedule_execution(int to_schedule, Scheduler *scheduler) {
    while (scheduled < to_schedule) {
#ifdef PARALLEL
        scheduler->schedule_execution(&this->cuda_stream, instructions[scheduled++]);

        // If necessary, schedule events
        for (int i = 0; i < IO_TYPE_SIZE; ++i)
            if (scheduled == last_index[i] + 1)
                scheduler->schedule_event(&this->cuda_stream, events[i]);
#else
        scheduler->schedule_execution(instructions[scheduled++]);
#endif
    }
}

void Stream::schedule_execution(Scheduler *scheduler) {
    this->schedule_execution(instructions.size(), scheduler);
}

void Stream::schedule_execution(IOType type, Scheduler *scheduler) {
    this->schedule_execution(last_index[type] + 1, scheduler);
}

void Stream::schedule_weight_update(Scheduler *scheduler) {
    for (int i = 0; i < instructions.size(); ++i)
#ifdef PARALLEL
        scheduler->schedule_weight_update(&this->cuda_stream, instructions[i]);
#else
        scheduler->schedule_weight_update(instructions[i]);
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
