#include <queue>
#include <set>

#include "engine/stream_cluster.h"

SequentialStreamCluster::SequentialStreamCluster(Model *model, State *state)
        : StreamCluster(model, state) {
    // Keep track of visited layers;
    std::set<Layer*> visited;
    std::set<Layer*> enqueued;

    // Create queue and push output layers
    std::queue<Layer*> queue;
    for (auto layer : model->get_layers(OUTPUT)) {
        queue.push(layer);
        enqueued.insert(layer);
    }
    for (auto layer : model->get_layers(INPUT_OUTPUT)) {
        queue.push(layer);
        enqueued.insert(layer);
    }

    /* Do breadth first search backwards on the model and create streams */
    while (not queue.empty()) {
        auto curr_layer = queue.front();
        queue.pop();
        visited.insert(curr_layer);

        // Add elements to beginning of list
#ifdef PARALLEL
        streams.insert(streams.begin(), new Stream(curr_layer, state, state->state_stream));
#else
        streams.insert(streams.begin(), new Stream(curr_layer, state));
#endif

        // Add any layers that feed into this one to if all of its
        //     output layers have been visited and its not in the queue
        for (auto conn : curr_layer->get_input_connections()) {
            if (visited.find(conn->from_layer) == visited.end()) {
                for (auto to_conn : conn->to_layer->get_output_connections())
                    if (visited.find(to_conn->to_layer) == visited.end())
                        continue;
                if (enqueued.find(conn->from_layer) == enqueued.end())
                    queue.push(conn->from_layer);
            }
        }
    }
}

SequentialStreamCluster::~SequentialStreamCluster() {
    // Delete streams
    for (auto stream : streams) delete stream;
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void SequentialStreamCluster::launch_post_input_calculations() {
#ifdef PARALLEL
    // Ensure all layer streams wait for input event
    // This function should not be called until this event has been
    //   scheduled to be recorded
    wait_event(this->state->input_event);
#endif
    // Activate streams
    for (auto it = streams.begin() ; it != streams.end(); ++it) {
        for (auto& inst : (*it)->get_instructions())
            inst->activate();
        state->update_states((*it)->to_layer);
    }
}

void SequentialStreamCluster::launch_weight_update() {
#ifdef PARALLEL
    wait_event(this->state->state_event);
#endif
    // Update streams backwards
    for (auto it = streams.rbegin() ; it != streams.rend(); ++it)
        for (auto& inst : (*it)->get_instructions())
            if (inst->is_plastic()) inst->update();
}

/******************************************************************************/
/**************************** EVENT HANDLING **********************************/
/******************************************************************************/

#ifdef PARALLEL
void SequentialStreamCluster::wait_event(cudaEvent_t *event) {
    cudaStreamWaitEvent(state->state_stream, *event, 0);
}
#endif
