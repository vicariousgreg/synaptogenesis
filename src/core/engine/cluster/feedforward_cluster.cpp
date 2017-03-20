#include <set>

#include "engine/cluster/cluster.h"

static bool DFS(Layer* curr_layer, std::set<Layer*>& visited) {
    if (visited.find(curr_layer) != visited.end()) {
        // If visited already, we found a cycle
        visited.clear();
        return false;
    } else {
        // Otherwise, push current layer and recurse
        visited.insert(curr_layer);
        // Be careful not to leave the structure
        for (auto& conn : curr_layer->get_output_connections())
            if (conn->to_layer->structure == curr_layer->structure
                and not DFS(conn->to_layer, visited)) return false;
        visited.erase(curr_layer);
        return true;
    }
}

FeedforwardCluster::FeedforwardCluster(Structure *structure,
        State *state, Environment *environment)
        : SequentialCluster(structure, state, environment) {
    // Determine if there are any cycles
    // Perform DFS on all input layers
    std::set<Layer*> visited;
    for (auto& layer : structure->get_layers())
        if (layer->is_input() and not DFS(layer, visited))
            ErrorManager::get_instance()->log_error(
                "Feedforward engine requires a structure with no cycles!");
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void FeedforwardCluster::launch_weight_update() {
    // Perform learning in reverse
    for (auto it = nodes.rbegin() ; it != nodes.rend() ; ++it)
        for (auto& inst : (*it)->get_update_instructions())
            inst->activate();
}
