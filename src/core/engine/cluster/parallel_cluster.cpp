#include <queue>

#include "engine/cluster/cluster.h"
#include "engine/instruction.h"
#include "model/structure.h"
#include "state/state.h"
#include "util/resource_manager.h"

ParallelCluster::ParallelCluster(Structure *structure,
        State *state, Environment *environment)
        : Cluster(state, environment) {
    // Build instructions
    for (auto& layer : structure->get_layers()) {
        auto device_id = state->get_device_id(layer);
        auto node = new ClusterNode(
            layer, state, environment, io_streams[device_id],
            ResourceManager::get_instance()->create_stream(device_id));
        nodes.push_back(node);
    }

    /* Schedule instructions */
    // Find all instructions with INPUT flag
    pre_input_instructions = sort_instructions(0, INPUT | EXPECTED, false);
    // Find all instructions without INPUT flag
    post_input_instructions = sort_instructions(INPUT | EXPECTED, 0, false);
    // Find all plastic instructions
    plastic_instructions = sort_instructions(0, 0, true);
}

InstructionList ParallelCluster::sort_instructions(
        IOTypeMask include, IOTypeMask exclude, bool plastic) {
    std::map<Layer*, std::queue<Instruction*> > schedules;
    InstructionList destination;

    // Extract instructions
    for (auto& node : nodes) {
        if ((include == 0 or (node->to_layer->get_type() & include)) and
                not (node->to_layer->get_type() & exclude)) {
            // Add to schedule map for round robin
            if (plastic) {
                for (auto& inst : node->get_update_instructions())
                    schedules[node->to_layer].push(inst);
            } else {
                for (auto& inst : node->get_activate_instructions())
                    schedules[node->to_layer].push(inst);
            }
        }
    }

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

/* The parallel cluster activates inter device transfers at the beginning
 *   of the cycle.  If it's a new instruction, copy over the dependencies
 *   from the synapse instruction and add it to the list.  Regardless of
 *   whether it's new, make the synapse instruction depend on it. */
void ParallelCluster::add_inter_device_instruction(
        Instruction *synapse_instruction,
        Instruction *inter_device_instruction,
        bool new_transfer) {
    // Add new transfers to the list to be executed
    // Copy the synapse instructions dependencies over
    if (new_transfer) {
        inter_device_instruction->copy_dependencies(synapse_instruction);
        inter_device_instructions.push_back(inter_device_instruction);
    }
    // Regardless of whether it's new, make the synapse instruction
    //   depend on the transfer instruction
    synapse_instruction->add_dependency(inter_device_instruction);
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void ParallelCluster::launch_pre_input_calculations() {
    for (auto& inst : this->inter_device_instructions) inst->activate();
    for (auto& inst : this->pre_input_instructions) inst->activate();
}

void ParallelCluster::launch_post_input_calculations() {
    for (auto& inst : this->post_input_instructions) inst->activate();
}

void ParallelCluster::launch_state_update() {
    for (auto& node : nodes) node->activate_state();
}

void ParallelCluster::launch_weight_update() {
    for (auto& inst : this->plastic_instructions) inst->activate();
}
