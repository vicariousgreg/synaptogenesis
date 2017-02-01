#include "engine/stream.h"

Stream::Stream(Layer *layer) : to_layer(layer), scheduled(0) {
    for (auto type : IOTypes)
        last_index[type] = 0;

#ifdef PARALLEL
    // Create cuda events
    cudaStreamCreate(&cuda_stream);
    finished_event = new cudaEvent_t;
    for (auto type : IOTypes)
        events[type] = new cudaEvent_t;

    cudaEventCreateWithFlags(finished_event, cudaEventDisableTiming);
    for (auto type : IOTypes)
        cudaEventCreateWithFlags(events[type], cudaEventDisableTiming);
#endif
}

Stream::~Stream() {
#ifdef PARALLEL
    for (auto type : IOTypes) {
        cudaEventDestroy(*events[type]);
        delete events[type];
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

void Stream::add_instruction(Instruction *inst) {
    this->instructions.push_back(inst);
#ifdef PARALLEL
    inst->set_stream(&this->cuda_stream);
#endif
}

void Stream::finalize() {
#ifdef PARALLEL
    // Add events for finishing each type of computation
    for (auto type : IOTypes)
        instructions[last_index[type]]->add_event(events[type]);

    // Add finished event
    instructions[instructions.size()-1]->add_event(finished_event);
#endif
}

void Stream::schedule(int to_schedule, InstructionList &schedule) {
    while (scheduled < to_schedule)
        schedule.push_back(instructions[scheduled++]);
}

void Stream::schedule(InstructionList &schedule) {
    this->schedule(instructions.size(), schedule);
}

void Stream::schedule(IOType type, InstructionList &schedule) {
    this->schedule(last_index[type] + 1, schedule);
}

void Stream::schedule_plastic(InstructionList &schedule) {
    for (auto inst : this->instructions)
        if (inst->is_plastic()) schedule.push_back(inst);
}

#ifdef PARALLEL
void Stream::wait_event(cudaEvent_t *event) {
    cudaStreamWaitEvent(this->cuda_stream, *event, 0);
}
#endif
