#include <queue>
#include <set>

#include "engine/cluster/cluster.h"
#include "engine/instruction.h"
#include "network/structure.h"
#include "state/state.h"
#include "util/resources/resource_manager.h"

SequentialCluster::SequentialCluster(Structure *structure,
    State *state, Engine *engine, PropertyConfig args)
        : Cluster(state, engine, args) {
    // Create compute streams
    auto res_man = ResourceManager::get_instance();
    for (auto device_id : state->get_active_devices())
        compute_streams[device_id] = res_man->create_stream(device_id);

    // Keep track of visited layers;
    std::set<Layer*> visited;

    // Create queue and push output layers
    // Add layers any layers that project to another structure
    //   or have no output connections
    std::queue<Layer*> queue;
    for (auto& layer : structure->get_layers())
        if (layer->is_structure_output()
                or layer->get_output_connections().size() == 0)
            queue.push(layer);

    /* Do breadth first search backwards on the network and create nodes */
    while (not queue.empty()) {
        auto curr_layer = queue.front();
        queue.pop();

        // If already visited, skip
        if (visited.find(curr_layer) != visited.end()) continue;

        // If current layer projects to other layers that haven't been visited,
        //   Come back later
        bool skip = false;
        for (auto& conn : curr_layer->get_output_connections())
            if (conn->to_layer != curr_layer
                    and conn->to_layer->structure == structure
                    and not conn->recurrent
                    and visited.find(conn->to_layer) == visited.end())
                skip = true;

        if (skip) {
            queue.push(curr_layer);
            continue;
        }

        visited.insert(curr_layer);

        // Add elements to beginning of list
        DeviceID device_id = state->get_device_id(curr_layer);
        nodes.insert(nodes.begin(),
            new ClusterNode(curr_layer, state, engine,
                io_streams[device_id], compute_streams[device_id]));

        // Add any other layers in the same structure that feed into 
        //   this one, and that have not been visited already.
        for (auto& conn : curr_layer->get_input_connections())
            if (conn->from_layer != curr_layer
                    and conn->from_layer->structure == structure
                    and not conn->recurrent
                    and visited.find(conn->from_layer) == visited.end())
                queue.push(conn->from_layer);
    }

    if (visited.size() < structure->get_layers().size())
        LOG_ERROR(
            "Sequential cluster failed to process all layers!");
}

/* Because the SequentialCluster computes nodes and synapses in order,
 *   this function simply adds the transfer instruction as a child for
 *   the first synapse instruction it showed up with, which should be
 *   passed in with |new_transfer| == true.  This synapse instruction
 *   will be executed before all others, so it's the only one that needs
 *   to be concerned about the transfer. */
void SequentialCluster::add_inter_device_instruction(
        Connection *conn,
        Instruction *synapse_instruction,
        Instruction *inter_device_instruction,
        bool new_transfer) {
    if (new_transfer) {
        // Add as child to synapse_instruction
        synapse_instruction->add_child(inter_device_instruction);
    } else {
        // Set the synapse instruction to depend on transfer
        // If the InterDevice instruction is created, this
        //   happens automatically in add_child()
        synapse_instruction->add_dependency(inter_device_instruction);
    }
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void SequentialCluster::launch_post_input_calculations() {
    // Activate nodes forwards
    for (auto it = nodes.begin() ; it != nodes.end(); ++it) {
        for (auto& inst : (*it)->get_activate_instructions())
            inst->activate();
        (*it)->activate_state();
    }
}

void SequentialCluster::launch_weight_update() {
    // Update nodes backwards
    for (auto it = nodes.rbegin() ; it != nodes.rend(); ++it)
        for (auto& inst : (*it)->get_update_instructions())
            inst->activate();
}
