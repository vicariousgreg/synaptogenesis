#include <set>

#include "engine/cluster/cluster.h"
#include "engine/instruction.h"
#include "network/structure.h"
#include "state/state.h"
#include "util/resource_manager.h"

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
        State *state, Engine *engine)
        : SequentialCluster(structure, state, engine) {
    // Determine if there are any cycles
    // Perform DFS on all input layers
    std::set<Layer*> visited;
    for (auto& layer : structure->get_layers())
        if (engine->is_input(layer) and not DFS(layer, visited))
            ErrorManager::get_instance()->log_error(
                "Feedforward engine requires a structure with no cycles!");
}
