#include <queue>
#include <set>

#include "engine/stream/cluster.h"

SequentialCluster::SequentialCluster(Structure *structure,
        State *state, Environment *environment)
        : Cluster(state, environment), compute_stream(new Stream()) {
    // Keep track of visited layers;
    std::set<Layer*> visited;

    // Create queue and push output layers
    std::queue<Layer*> queue;
    for (auto layer : structure->get_layers())
        if (layer->is_output()) queue.push(layer);

    /* Do breadth first search backwards on the model and create nodes */
    while (not queue.empty()) {
        auto curr_layer = queue.front();
        queue.pop();

        // If already visited, skip
        if (visited.find(curr_layer) != visited.end()) continue;
        visited.insert(curr_layer);

        // Add elements to beginning of list
        nodes.insert(nodes.begin(),
            new ClusterNode(curr_layer, state, environment, compute_stream));

        // Add any layers that feed into this one to if all of its
        //     output layers have been visited and its not in the queue
        // Also ensure that the traversal does not leave the structure
        for (auto conn : curr_layer->get_input_connections()) {
            if (visited.find(conn->from_layer) == visited.end()) {
                for (auto to_conn : conn->to_layer->get_output_connections())
                    if (visited.find(to_conn->to_layer) == visited.end()
                        or to_conn->to_layer->structure != structure)
                        continue;
                if (conn->from_layer->structure == structure)
                    queue.push(conn->from_layer);
            }
        }
    }
}

SequentialCluster::~SequentialCluster() {
    // Delete nodes
    for (auto node : nodes) delete node;
    delete compute_stream;
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void SequentialCluster::launch_input() {
    for (auto& node : nodes)
        node->activate_input_instruction();
}

void SequentialCluster::launch_output() {
    for (auto& node : nodes)
        node->activate_output_instruction();
}

void SequentialCluster::launch_post_input_calculations() {
    // Activate nodes
    for (auto it = nodes.begin() ; it != nodes.end(); ++it) {
        for (auto& inst : (*it)->get_instructions())
            inst->activate();
        (*it)->activate_state_instruction();
    }
}

void SequentialCluster::launch_weight_update() {
    // Update nodes backwards
    for (auto it = nodes.rbegin() ; it != nodes.rend(); ++it)
        for (auto& inst : (*it)->get_instructions())
            if (inst->is_plastic()) inst->update();
}

/******************************************************************************/
/**************************** EVENT HANDLING **********************************/
/******************************************************************************/

void SequentialCluster::wait_for_input() {
    for (auto node : nodes)
        node->get_input_event()->synchronize();
}
void SequentialCluster::wait_for_output() {
    for (auto node : nodes)
        node->get_output_event()->synchronize();
}