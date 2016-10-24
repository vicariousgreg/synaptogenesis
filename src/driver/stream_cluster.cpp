#include <vector>
#include "driver/stream_cluster.h"
#include "driver/driver.h"

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
