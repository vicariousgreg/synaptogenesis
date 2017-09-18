#include <queue>
#include <set>

#include "engine/cluster/cluster.h"
#include "engine/instruction.h"
#include "network/structure.h"
#include "state/state.h"
#include "util/resource_manager.h"

SequentialCluster::SequentialCluster(Structure *structure,
        State *state, Engine *engine)
        : Cluster(state, engine) {
    // Create compute streams
    auto res_man = ResourceManager::get_instance();
    for (DeviceID i = 0 ; i < res_man->get_num_devices(); ++i)
        compute_streams.push_back(res_man->create_stream(i));

    // Keep track of visited layers;
    std::set<Layer*> visited;

    // Create queue and push output layers
    // Add formal output layers, and any layers that project
    //   to another structure
    std::queue<Layer*> queue;
    for (auto& layer : structure->get_layers())
        if (engine->is_output(layer)) queue.push(layer);
        else
            for (auto& conn : layer->get_output_connections())
                if (conn->to_layer->structure != structure)
                    queue.push(layer);

    /* Do breadth first search backwards on the network and create nodes */
    while (not queue.empty()) {
        auto curr_layer = queue.front();
        queue.pop();

        // If already visited, skip
        if (visited.find(curr_layer) != visited.end()) continue;
        visited.insert(curr_layer);

        // Add elements to beginning of list
        DeviceID device_id = state->get_device_id(curr_layer);
        nodes.insert(nodes.begin(),
            new ClusterNode(curr_layer, state, engine,
                io_streams[device_id], compute_streams[device_id]));

        // Add any layers that feed into this one to if all of its
        //     output layers have been visited and its not in the queue
        // Also ensure that the traversal does not leave the structure
        for (auto& conn : curr_layer->get_input_connections()) {
            if (visited.find(conn->from_layer) == visited.end()) {
                for (auto& to_conn : conn->to_layer->get_output_connections())
                    if (visited.find(to_conn->to_layer) == visited.end()
                        or to_conn->to_layer->structure != structure)
                        continue;
                if (conn->from_layer->structure == structure)
                    queue.push(conn->from_layer);
            }
        }
    }

    if (visited.size() < structure->get_layers().size())
        ErrorManager::get_instance()->log_error(
            "Sequential cluster failed to process all layers!");
}

/* Because the SequentialCluster computes nodes and synapses in order,
 *   this function simply adds the transfer instruction as a child for
 *   the first synapse instruction it showed up with, which should be
 *   passed in with |new_transfer| == true.  This synapse instruction
 *   will be executed before all others, so it's the only one that needs
 *   to be concerned about the transfer. */
void SequentialCluster::add_inter_device_instruction(
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
