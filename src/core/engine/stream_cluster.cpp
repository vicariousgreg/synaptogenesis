#include "engine/stream_cluster.h"

StreamCluster::StreamCluster(Model *model, State *state)
        : state(state) {
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

        Stream *stream =
            (it != streams[to_layer->type].end())
            ? it->second : new Stream(to_layer);

        stream->add_instruction(inst, from_layer->type);
        streams[to_layer->type][to_layer] = stream;

        // Add to global list
        all_instructions.push_back(inst);
    }

    // Finalize all streams
    // This sets up cuda information
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto& stream : streams[i]) stream.second->finalize();

    // Schedule clear output instructions
    schedule_to(OUTPUT);
    this->sort_schedule(this->clear_output_instructions);

    // Schedule input output instructions
    schedule_to(INPUT_OUTPUT);
    schedule_from(INPUT_OUTPUT);
    schedule_from(OUTPUT);
    this->sort_schedule(this->input_output_instructions);

    // Schedule non output instructions
    schedule_to(INPUT);
    schedule_to(INTERNAL);
    this->sort_schedule(this->non_output_instructions);

    // Schedule plastic
    schedule_plastic();
    this->sort_schedule(this->plastic_instructions);
}

StreamCluster::~StreamCluster() {
    // Delete streams
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto& stream : streams[i]) delete stream.second;

    // Delete instructions
    for (auto& inst : this->all_instructions) delete inst;
}

void StreamCluster::disable_learning() {
    for (int i = 0; i < all_instructions.size(); ++i)
        all_instructions[i]->disable_learning();
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void StreamCluster::launch_clear_output_calculations() {
#ifdef PARALLEL
    // Ensure all layer streams wait for clear event
    wait_event(OUTPUT, this->state->clear_event);
    wait_event(INTERNAL, this->state->clear_event);
#endif

    // Launch clear output relevant computations
    for (auto& inst : this->clear_output_instructions) inst->activate();
}

void StreamCluster::launch_input_output_calculations() {
#ifdef PARALLEL
    // Ensure all layer streams wait for input event
    // This function should not be called until this event has been
    //   scheduled to be recorded
    wait_event(INPUT, this->state->input_event);
    wait_event(INPUT_OUTPUT, this->state->input_event);
#endif

    // Launch input output relevant computations
    for (auto& inst : this->input_output_instructions) inst->activate();

#ifdef PARALLEL
    block_stream_from(OUTPUT, state->state_stream);
    block_stream_from(INPUT_OUTPUT, state->state_stream);
    block_stream_to(INPUT_OUTPUT, state->state_stream);
    block_stream_to(OUTPUT, state->state_stream);
#endif
}

void StreamCluster::launch_non_output_calculations() {
    // Launch output relevant computations
    for (auto& inst : this->non_output_instructions) inst->activate();

#ifdef PARALLEL
    block_stream_to(INPUT, state->state_stream);
    block_stream_to(INTERNAL, state->state_stream);
#endif
}

void StreamCluster::launch_weight_update() {
    for (auto& inst : this->plastic_instructions) inst->update();
}

/******************************************************************************/
/***************************** SCHEDULERS *************************************/
/******************************************************************************/

void StreamCluster::schedule_from(IOType from_type) {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto& stream : streams[i])
            stream.second->schedule(from_type,
                this->schedules[stream.second->to_layer]);
}

void StreamCluster::schedule_to(IOType to_type) {
    for (auto& stream : streams[to_type])
        stream.second->schedule(
            this->schedules[stream.second->to_layer]);
}

void StreamCluster::schedule_plastic() {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto& stream : streams[i])
            stream.second->schedule_plastic(
                this->schedules[stream.second->to_layer]);
}

void StreamCluster::sort_schedule(InstructionList &destination) {
    // Perform round robin on streams
    bool done = false;
    for (int i = 0; not done; ++i) {
        done = true;
        for (auto& schedule : this->schedules) {
            if (i < schedule.second.size()) {
                done = false;
                destination.push_back(schedule.second[i]);
            }
        }
    }

    // Clear schedule
    for (auto& schedule : this->schedules)
        schedule.second.clear();
}

/******************************************************************************/
/**************************** EVENT HANDLING **********************************/
/******************************************************************************/

#ifdef PARALLEL
void StreamCluster::wait_event(IOType to_type, cudaEvent_t *event) {
    for (auto& stream : streams[to_type])
        stream.second->wait_event(event);
}

void StreamCluster::block_stream_to(IOType to_type, cudaStream_t cuda_stream) {
    for (auto& stream : streams[to_type])
        cudaStreamWaitEvent(cuda_stream, *stream.second->finished_event, 0);
}

void StreamCluster::block_stream_from(IOType from_type, cudaStream_t cuda_stream) {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        for (auto& stream : streams[i])
            cudaStreamWaitEvent(cuda_stream, *stream.second->events[from_type], 0);
}
#endif
