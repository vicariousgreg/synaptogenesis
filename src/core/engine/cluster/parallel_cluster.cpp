#include <queue>

#include "engine/cluster/cluster.h"
#include "engine/instruction.h"
#include "network/structure.h"
#include "state/state.h"
#include "util/resources/resource_manager.h"

static InstructionList round_robin(
        std::map<Layer*, std::queue<Instruction*>> schedules) {
    InstructionList destination;

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

ParallelCluster::ParallelCluster(Structure *structure,
    State *state, Engine *engine, PropertyConfig args)
        : Cluster(state, engine, args) {
    auto res_man = ResourceManager::get_instance();

    // Build instructions
    for (auto& layer : structure->get_layers()) {
        auto device_id = state->get_device_id(layer);
        auto node = new ClusterNode(
            layer, state, engine, io_streams[device_id],
            res_man->create_stream(device_id));
        nodes.push_back(node);
    }

    /* Schedule instructions */
    std::map<Layer*, std::queue<Instruction*>> pre_input_schedules;
    std::map<Layer*, std::queue<Instruction*>> post_input_schedules;
    std::map<Layer*, std::queue<Instruction*>> plastic_schedules;

    // Extract instructions
    for (auto& node : nodes) {
        // Find all instructions with INPUT flag
        if (engine->get_io_type(node->to_layer) & INPUT) {
            for (auto& inst : node->get_activate_instructions())
                post_input_schedules[node->to_layer].push(inst);
        // Find all instructions without INPUT flag
        } else {
            auto insts = node->get_activate_instructions();
            int ghost_inst_index = node->get_ghost_inst_index();

            // Add pre-ghost instructions
            for (auto& inst : InstructionList(insts.begin(), insts.begin() + ghost_inst_index))
                pre_input_schedules[node->to_layer].push(inst);

            // Add post-ghost instructions
            for (auto& inst : InstructionList(insts.begin() + ghost_inst_index, insts.end()))
                post_input_schedules[node->to_layer].push(inst);
        }

        // Find all plastic instructions
        for (auto& inst : node->get_update_instructions())
            plastic_schedules[node->to_layer].push(inst);
    }

    pre_input_instructions = round_robin(pre_input_schedules);
    post_input_instructions = round_robin(post_input_schedules);
    plastic_instructions = round_robin(plastic_schedules);
}

/* The parallel cluster activates inter device transfers at the beginning
 *   of the cycle.  If it's a new instruction, copy over the dependencies
 *   from the synapse instruction and add it to the list.  Regardless of
 *   whether it's new, make the synapse instruction depend on it. */
void ParallelCluster::add_inter_device_instruction(
        Connection *conn,
        Instruction *synapse_instruction,
        Instruction *inter_device_instruction,
        bool new_transfer) {
    // Add new transfers to the list to be executed
    // Copy the synapse instructions dependencies over
    if (new_transfer) {
        inter_device_instruction->copy_dependencies(synapse_instruction);

        // Transfers from ghost layers must wait for input
        if (conn->from_layer->is_ghost)
            post_input_inter_device_instructions.push_back(
                inter_device_instruction);
        else
            pre_input_inter_device_instructions.push_back(
                inter_device_instruction);
    }
    // Regardless of whether it's new, make the synapse instruction
    //   depend on the transfer instruction
    synapse_instruction->add_dependency(inter_device_instruction);
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void ParallelCluster::launch_pre_input_calculations() {
    for (auto& inst : this->pre_input_inter_device_instructions) inst->activate();
    for (auto& inst : this->pre_input_instructions) inst->activate();
}

void ParallelCluster::launch_post_input_calculations() {
    // Compute ghost output before other layers use it
    for (auto& node : nodes)
        if (node->to_layer->is_ghost)
            node->activate_state();

    for (auto& inst : this->post_input_inter_device_instructions) inst->activate();
    for (auto& inst : this->post_input_instructions) inst->activate();
}

void ParallelCluster::launch_state_update() {
    // Skip ghost output
    for (auto& node : nodes)
        if (not node->to_layer->is_ghost)
            node->activate_state();
}

void ParallelCluster::launch_weight_update() {
    for (auto& inst : this->plastic_instructions) inst->activate();
}
