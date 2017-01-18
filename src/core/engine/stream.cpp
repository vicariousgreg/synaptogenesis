#include "engine/stream.h"
#include "engine/stream_cluster.h"

Stream::Stream(Layer *layer) : to_layer(layer), scheduled(0) {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        last_index[i] = 0;
#ifdef PARALLEL
    // Create cuda events
    cudaStreamCreate(&cuda_stream);
    finished_event = new cudaEvent_t;
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        events[i] = new cudaEvent_t;

    cudaEventCreateWithFlags(finished_event, cudaEventDisableTiming);
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        cudaEventCreateWithFlags(events[i], cudaEventDisableTiming);
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
}

void Stream::schedule(int to_schedule, StreamCluster *stream_cluster) {
    while (scheduled < to_schedule)
        stream_cluster->schedule(instructions[scheduled++]);
}

void Stream::schedule(StreamCluster *stream_cluster) {
    this->schedule(instructions.size(), stream_cluster);
}

void Stream::schedule(IOType type, StreamCluster *stream_cluster) {
    this->schedule(last_index[type] + 1, stream_cluster);
}

void Stream::schedule_plastic(StreamCluster *stream_cluster) {
    for (auto inst : this->instructions)
        if (inst->is_plastic())
            stream_cluster->schedule(inst);
}

#ifdef PARALLEL
void Stream::wait_event(cudaEvent_t *event) {
    cudaStreamWaitEvent(this->cuda_stream, *event, 0);
}
#endif
