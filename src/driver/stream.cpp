#include <vector>
#include "driver/stream.h"
#include "driver/driver.h"

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
        for (int i = 0; i < IO_TYPE_SIZE; ++i)
            if (scheduled == last_index[i] + 1)
                cudaEventRecord(*events[i], cuda_stream);
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

StreamCluster::StreamCluster(Model *model, State *state)
        : state(state), scheduler(new Scheduler) {
    // Build instructions
    for (int i = 0; i < model->connections.size(); ++i) {
        Connection *conn = model->connections[i];
        Layer *from_layer = conn->from_layer;
        Layer *to_layer = conn->to_layer;

        // Create instruction
        Instruction *inst = new Instruction(conn, state);

        // Add instruction to appropriate stream
        std::map<Layer*, Stream*>::iterator it =
            streams[to_layer->type].find(to_layer);
        Stream *stream;
        if (it != streams[to_layer->type].end()) {
            stream = it->second;
        } else {
            stream = new Stream(to_layer);
        }
        stream->add_instruction(inst, from_layer->type);
        streams[to_layer->type][to_layer] = stream;
    }
}

StreamCluster::~StreamCluster() {
    delete scheduler;
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto it = streams[i].begin(); it != streams[i].end(); ++it)
            delete it->second;
}

void StreamCluster::schedule_output_calculations() {
    // Launch output relevant computations
    schedule_execution_from(INPUT_OUTPUT);
    schedule_execution_from(OUTPUT);
    schedule_execution_to(OUTPUT);
    schedule_execution_to(INPUT_OUTPUT);
}

void StreamCluster::schedule_non_output_calculations() {
    // Launch output relevant computations
    schedule_execution_to(INPUT);
    schedule_execution_to(INTERNAL);
}

void StreamCluster::reset() {
    // Reset streams
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto it = streams[i].begin(); it != streams[i].end(); ++it)
            it->second->reset();

#ifdef PARALLEL
    // Ensure all layer streams wait for appropriate input
    wait_event(INPUT, this->state->input_event);
    wait_event(INPUT_OUTPUT, this->state->input_event);
    wait_event(OUTPUT, this->state->clear_event);
    wait_event(INTERNAL, this->state->clear_event);
#endif
}

void StreamCluster::schedule_execution_from(IOType from_type) {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto it = streams[i].begin(); it != streams[i].end(); ++it)
            it->second->schedule_execution(from_type, scheduler);
}

void StreamCluster::schedule_execution_to(IOType to_type) {
    for (auto it = streams[to_type].begin(); it != streams[to_type].end(); ++it)
        it->second->schedule_execution(scheduler);
}

void StreamCluster::schedule_weight_update() {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto it = streams[i].begin(); it != streams[i].end(); ++it)
            it->second->schedule_weight_update(scheduler);
}

void StreamCluster::dispatch(Driver *driver) {
    scheduler->dispatch(driver);
}

bool StreamCluster::is_done() {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto it = streams[i].begin(); it != streams[i].end(); ++it)
            if (not it->second->is_done())
                return false;
    return true;
}

bool StreamCluster::is_done(IOType type) {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto it = streams[i].begin(); it != streams[i].end(); ++it)
            if (not it->second->is_done(type))
                return false;
    return true;
}

#ifdef PARALLEL
void StreamCluster::wait_event(IOType to_type, cudaEvent_t *event) {
    for (auto it = streams[to_type].begin(); it != streams[to_type].end(); ++it)
        it->second->wait_event(event);
}

void StreamCluster::block_stream_to(IOType to_type, cudaStream_t stream) {
    for (auto it = streams[to_type].begin(); it != streams[to_type].end(); ++it)
        cudaStreamWaitEvent(stream, *it->second->finished_event, 0);
}

void StreamCluster::block_stream_from(IOType from_type, cudaStream_t stream) {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto it = streams[i].begin(); it != streams[i].end(); ++it)
            cudaStreamWaitEvent(stream, *it->second->events[from_type], 0);
}

void StreamCluster::block_state_on_output_calculations() {
    block_stream_from(OUTPUT, state->state_stream);
    block_stream_from(INPUT_OUTPUT, state->state_stream);
    block_stream_to(INPUT_OUTPUT, state->state_stream);
    block_stream_to(OUTPUT, state->state_stream);
}

void StreamCluster::block_state_on_non_output_calculations() {
    block_stream_to(INPUT, this->state->state_stream);
    block_stream_to(INTERNAL, this->state->state_stream);
}
#endif
