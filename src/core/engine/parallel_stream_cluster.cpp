#include <queue>

#include "engine/stream_cluster.h"

ParallelStreamCluster::ParallelStreamCluster(Structure *structure,
        State *state, Environment *environment)
        : StreamCluster(state, environment) {
    // Build instructions
    for (auto& layer : structure->get_layers())
        streams[layer->get_type()].push_back(
            new Stream(layer, state, environment));

    // Schedule instructions
    post_input_instructions = sort_instructions(
        IOTypeVector { INPUT, INPUT_OUTPUT },
        false);
    pre_input_instructions = sort_instructions(
        IOTypeVector { OUTPUT, INTERNAL },
        false);
    plastic_instructions = sort_instructions(
        IOTypeVector { INPUT, INPUT_OUTPUT, OUTPUT, INTERNAL },
        true);
}

ParallelStreamCluster::~ParallelStreamCluster() {
    // Delete streams
    for (auto type : IOTypes)
        for (auto stream : streams[type]) delete stream;
}

InstructionList ParallelStreamCluster::sort_instructions(
        IOTypeVector types, bool plastic) {
    std::map<Layer*, std::queue<Instruction* > > schedules;
    InstructionList destination;

    // Extract instructions
    for (auto type : types)
        for (auto& stream : streams[type])
            for (auto& inst : stream->get_instructions())
                if (not plastic or inst->is_plastic())
                    // Add to schedule map for round robin
                    schedules[stream->to_layer].push(inst);

    // Perform round robin on streams
    // Connections should be initialized this way to take advantage of
    //   stream overlap
    while (schedules.size() > 0) {
        for (auto it = schedules.begin(); it != schedules.end(); ) {
            destination.push_back(it->second.front());
            it->second.pop();
            if (it->second.size() == 0) it = schedules.erase(it);
            else ++it;
        }
    }
    return destination;
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void ParallelStreamCluster::launch_pre_input_calculations() {
    for (auto& inst : this->pre_input_instructions) inst->activate();
}

void ParallelStreamCluster::launch_input() {
    for (auto& stream : streams[INPUT])
        stream->activate_input_instruction();
    for (auto& stream : streams[INPUT_OUTPUT])
        stream->activate_input_instruction();
}

void ParallelStreamCluster::launch_post_input_calculations() {
    for (auto& inst : this->post_input_instructions) inst->activate();
}

void ParallelStreamCluster::launch_output() {
    for (auto& stream : streams[INPUT_OUTPUT])
        stream->activate_output_instruction();
    for (auto& stream : streams[OUTPUT])
        stream->activate_output_instruction();
}

void ParallelStreamCluster::launch_state_update() {
    for (auto type : IOTypes)
        for (auto& stream : streams[type])
            stream->activate_state_instruction();
}

void ParallelStreamCluster::launch_weight_update() {
    for (auto& inst : this->plastic_instructions) inst->update();
}

/******************************************************************************/
/**************************** EVENT HANDLING **********************************/
/******************************************************************************/

#ifdef PARALLEL
void ParallelStreamCluster::wait_for_input() {
    for (auto& stream : streams[INPUT])
        cudaEventSynchronize(stream->get_input_event());
    for (auto& stream : streams[INPUT_OUTPUT])
        cudaEventSynchronize(stream->get_input_event());
}

void ParallelStreamCluster::wait_for_output() {
    for (auto& stream : streams[INPUT_OUTPUT])
        cudaEventSynchronize(stream->get_output_event());
    for (auto& stream : streams[OUTPUT])
        cudaEventSynchronize(stream->get_output_event());
}
#endif
