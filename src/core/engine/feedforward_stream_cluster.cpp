#include <set>

#include "engine/stream_cluster.h"

static bool DFS(Layer* curr_layer, std::set<Layer*>& visited) {
    if (visited.find(curr_layer) != visited.end()) {
        // If visited already, we found a cycle
        visited.clear();
        return false;
    } else {
        // Otherwise, push current layer and recurse
        visited.insert(curr_layer);
        for (auto conn : curr_layer->get_output_connections())
            if (not DFS(conn->to_layer, visited)) return false;
        visited.erase(curr_layer);
        return true;
    }
}

FeedforwardStreamCluster::FeedforwardStreamCluster(Model *model, State *state)
        : SequentialStreamCluster(model, state) {
    // Determine if there are any cycles
    // Perform DFS on all input layers
    std::set<Layer*> visited;
    for (auto layer : model->get_layers(INPUT))
        if (not DFS(layer, visited))
            ErrorManager::get_instance()->log_error(
                "Feedforward engine requires a model with no cycles!");
    for (auto layer : model->get_layers(INPUT_OUTPUT))
        if (not DFS(layer, visited))
            ErrorManager::get_instance()->log_error(
                "Feedforward engine requires a model with no cycles!");
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void FeedforwardStreamCluster::launch_weight_update() {
#ifdef PARALLEL
    wait_event(this->state->state_event);
#endif
    // Perform learning in reverse
    for (auto it = streams.rbegin() ; it != streams.rend() ; ++it)
        for (auto& inst : (*it)->get_instructions())
            if (inst->is_plastic()) inst->update();
}
