#include <queue>
#include <set>

#include "engine/feedforward_stream_cluster.h"

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
    /* First, determine if there are any cycles */
    std::set<Layer*> visited;

    // Perform DFS on all input layers
    for (auto layer : model->get_layers(INPUT))
        if (not DFS(layer, visited))
            ErrorManager::get_instance()->log_error(
                "Feedforward engine requires a model with no cycles!");
    for (auto layer : model->get_layers(INPUT_OUTPUT))
        if (not DFS(layer, visited))
            ErrorManager::get_instance()->log_error(
                "Feedforward engine requires a model with no cycles!");

    // Create queue and push input layers
    std::queue<Layer*> queue;
    for (auto layer : model->get_layers(INPUT))
        queue.push(layer);
    for (auto layer : model->get_layers(INPUT_OUTPUT))
        queue.push(layer);

    /* Do breadth first search on the model and create streams */
    while (not queue.empty()) {
        auto curr_layer = queue.front();
        queue.pop();
        visited.insert(curr_layer);
#ifdef PARALLEL
        streams.push_back(new Stream(curr_layer, state, state->state_stream));
#else
        streams.push_back(new Stream(curr_layer, state));
#endif

        // Add any layers that this one outputs to if all of its
        //     input layers have been visited
        for (auto conn : curr_layer->get_output_connections()) {
            if (visited.find(conn->to_layer) == visited.end()) {
                for (auto from_conn : conn->to_layer->get_input_connections())
                    if (visited.find(from_conn->from_layer) == visited.end())
                        continue;
                queue.push(conn->to_layer);
            }
        }
    }
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
