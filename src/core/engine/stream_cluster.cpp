#include "engine/stream_cluster.h"

StreamCluster::StreamCluster(Model *model, State *state)
        : state(state) {
    // Build instructions
    // For each IO type...
    for (auto to_type : IOTypes) {
        // For each layer...
        for (auto& to_layer : model->get_layers(to_type)) {
            // Skip layers with no input connections (typically sensory)
            if (to_layer->get_input_connections().size() > 0) {
                Stream *stream = new Stream(to_layer);

                // Beform DFS on dendritic tree
                this->dendrite_DFS(to_layer->dendritic_root, stream);

                streams[to_type][to_layer] = stream;
                stream->finalize();
            }
        }
    }

    // Schedule input instructions
    schedule_to(INPUT);
    schedule_to(INPUT_OUTPUT);
    this->sort_schedule(this->input_instructions);

    // Schedule non input instructions
    schedule_to(OUTPUT);
    schedule_to(INTERNAL);
    this->sort_schedule(this->non_input_instructions);

    // Schedule plastic
    schedule_plastic();
    this->sort_schedule(this->plastic_instructions);
}

StreamCluster::~StreamCluster() {
    // Delete streams
    for (auto type : IOTypes)
        for (auto& stream : streams[type])
            delete stream.second;

    // Delete instructions
    for (auto& inst : this->all_instructions) delete inst;
}

void StreamCluster::dendrite_DFS(DendriticNode *curr, Stream *stream) {
    for (auto& child : curr->get_children()) {
        Instruction *inst;

        if (child->is_leaf()) { // Leaf node
            Connection* conn = child->conn;
            inst = new SynapseInstruction(conn, this->state);
            stream->add_instruction(inst, conn->from_layer->get_type());
        } else {            // Internal node
            this->dendrite_DFS(child, stream);
            inst = new DendriticInstruction(curr, child, state);
            stream->add_instruction(inst);
        }

        // Add to global list
        this->all_instructions.push_back(inst);
    }
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void StreamCluster::launch_non_input_calculations() {
#ifdef PARALLEL
    // Ensure all layer streams wait for clear event
    wait_event(OUTPUT, this->state->clear_event);
    wait_event(INTERNAL, this->state->clear_event);
#endif

    // Launch clear output relevant computations
    for (auto& inst : this->non_input_instructions) inst->activate();
}

void StreamCluster::launch_input_calculations() {
#ifdef PARALLEL
    // Ensure all layer streams wait for input event
    // This function should not be called until this event has been
    //   scheduled to be recorded
    wait_event(INPUT, this->state->input_event);
    wait_event(INPUT_OUTPUT, this->state->input_event);
#endif

    // Launch input output relevant computations
    for (auto& inst : this->input_instructions) inst->activate();

#ifdef PARALLEL
    block_stream_to(INPUT, state->state_stream);
    block_stream_to(INPUT_OUTPUT, state->state_stream);
    block_stream_to(OUTPUT, state->state_stream);
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
    for (auto type : IOTypes)
        for (auto& stream : streams[type])
            stream.second->schedule(from_type,
                this->schedules[stream.second->to_layer]);
}

void StreamCluster::schedule_to(IOType to_type) {
    for (auto& stream : streams[to_type])
        stream.second->schedule(
            this->schedules[stream.second->to_layer]);
}

void StreamCluster::schedule_plastic() {
    for (auto type : IOTypes)
        for (auto& stream : streams[type])
            stream.second->schedule_plastic(
                this->schedules[stream.second->to_layer]);
}

void StreamCluster::sort_schedule(InstructionList &destination) {
#ifdef PARALLEL
    // Perform round robin on streams
    // Connections should be initialized this way to take advantage of
    //   stream overlap
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
#else
    // Copy over to schedule
    for (auto& schedule : this->schedules)
        for (int i = 0; i < schedule.second.size(); ++i)
            destination.push_back(schedule.second[i]);
#endif

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
        cudaStreamWaitEvent(cuda_stream, *stream.second->get_finished_event(), 0);
}
#endif
