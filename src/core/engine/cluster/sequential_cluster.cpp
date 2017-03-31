#include <queue>
#include <set>

#include "engine/cluster/cluster.h"
#include "engine/instruction.h"
#include "model/structure.h"
#include "state/state.h"
#include "util/resource_manager.h"

SequentialCluster::SequentialCluster(Structure *structure,
        State *state, Environment *environment)
        : Cluster(state, environment) {
    // Create compute streams
    auto res_man = ResourceManager::get_instance();
    for (DeviceID i = 0 ; i < res_man->get_num_devices(); ++i)
        compute_streams.push_back(res_man->create_stream(i));

    // Keep track of visited layers;
    std::set<Layer*> visited;

    // Create queue and push output layers
    std::queue<Layer*> queue;
    for (auto& layer : structure->get_layers())
        if (layer->is_output()) queue.push(layer);

    /* Do breadth first search backwards on the model and create nodes */
    while (not queue.empty()) {
        auto curr_layer = queue.front();
        queue.pop();

        // If already visited, skip
        if (visited.find(curr_layer) != visited.end()) continue;
        visited.insert(curr_layer);

        // Add elements to beginning of list
        DeviceID device_id = state->get_device_id(curr_layer);
        nodes.insert(nodes.begin(),
            new ClusterNode(curr_layer, state, environment,
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
}

void SequentialCluster::add_external_dependencies(
        std::map<Layer*, ClusterNode*> all_nodes) {
    // Crawl through the nodes and add dependencies transfers
    // This prevents race conditions from output updates
    // Ensure that the latest update is available for this timestep
    // This is the opposite order of the parallel cluster
    for (auto& node : nodes)
        for (auto& pair : node->get_synapse_instructions())
            pair.second->add_dependency(
                all_nodes[pair.first->from_layer]->get_state_update_instruction());
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
