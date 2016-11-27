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
#ifdef PARALLEL
    for (int i = 0; i < IO_TYPE_SIZE; ++i) {
        cudaEventDestroy(*events[i]);
        delete events[i];
    }
    cudaEventDestroy(*finished_event);
    delete finished_event;
#endif
}

void Stream::add_instruction(Instruction *inst, IOType from_type) {
    this->last_index[from_type] = instructions.size();
    this->instructions.push_back(inst);
#ifdef PARALLEL
    inst->set_stream(&this->cuda_stream);
#endif
}

void Stream::finalize() {
#ifdef PARALLEL
    // Add events for finishing each type of computation
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        instructions[last_index[i]]->add_event(events[i]);

    // Add finished event
    instructions[instructions.size()-1]->add_event(finished_event);
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

void Stream::schedule(int to_schedule, Scheduler *scheduler) {
    while (scheduled < to_schedule)
        scheduler->schedule(instructions[scheduled++]);
}

void Stream::schedule(Scheduler *scheduler) {
    this->schedule(instructions.size(), scheduler);
}

void Stream::schedule(IOType type, Scheduler *scheduler) {
    this->schedule(last_index[type] + 1, scheduler);
}

void Stream::schedule_plastic(Scheduler *scheduler) {
    for (auto inst : this->instructions)
        if (inst->is_plastic())
            scheduler->schedule(inst);
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
