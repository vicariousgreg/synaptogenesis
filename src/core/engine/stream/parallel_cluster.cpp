#include <queue>

#include "engine/stream/cluster.h"

ParallelCluster::ParallelCluster(Structure *structure,
        State *state, Environment *environment)
        : Cluster(state, environment) {
    // Build instructions
    for (auto& layer : structure->get_layers())
        nodes[layer->get_type()].push_back(
            new ClusterNode(layer, state, environment));

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

ParallelCluster::~ParallelCluster() {
    // Delete nodes
    for (auto type : IOTypes)
        for (auto node : nodes[type]) delete node;
}

InstructionList ParallelCluster::sort_instructions(
        IOTypeVector types, bool plastic) {
    std::map<Layer*, std::queue<Instruction* > > schedules;
    InstructionList destination;

    // Extract instructions
    for (auto type : types)
        for (auto& node : nodes[type])
            for (auto& inst : node->get_instructions())
                if (not plastic or inst->is_plastic())
                    // Add to schedule map for round robin
                    schedules[node->to_layer].push(inst);

    // Perform round robin on nodes
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

void ParallelCluster::launch_pre_input_calculations() {
    for (auto& inst : this->pre_input_instructions) inst->activate();
}

void ParallelCluster::launch_input() {
    for (auto& node : nodes[INPUT])
        node->activate_input_instruction();
    for (auto& node : nodes[INPUT_OUTPUT])
        node->activate_input_instruction();
}

void ParallelCluster::launch_post_input_calculations() {
    for (auto& inst : this->post_input_instructions) inst->activate();
}

void ParallelCluster::launch_output() {
    for (auto& node : nodes[INPUT_OUTPUT])
        node->activate_output_instruction();
    for (auto& node : nodes[OUTPUT])
        node->activate_output_instruction();
}

void ParallelCluster::launch_state_update() {
    for (auto type : IOTypes)
        for (auto& node : nodes[type])
            node->activate_state_instruction();
}

void ParallelCluster::launch_weight_update() {
    for (auto& inst : this->plastic_instructions) inst->update();
}

/******************************************************************************/
/**************************** EVENT HANDLING **********************************/
/******************************************************************************/

void ParallelCluster::wait_for_input() {
    for (auto& node : nodes[INPUT])
        node->get_input_event()->synchronize();
    for (auto& node : nodes[INPUT_OUTPUT])
        node->get_input_event()->synchronize();
}

void ParallelCluster::wait_for_output() {
    for (auto& node : nodes[INPUT_OUTPUT])
        node->get_output_event()->synchronize();
    for (auto& node : nodes[OUTPUT])
        node->get_output_event()->synchronize();
}
