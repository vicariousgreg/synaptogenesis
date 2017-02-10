#include <queue>
#include <set>

#include "engine/sequential_stream_cluster.h"

SequentialStreamCluster::SequentialStreamCluster(Model *model, State *state)
        : state(state) {
    // Create queue and push input layers
    std::queue<Layer*> queue;
    for (auto layer : model->get_layers(INPUT))
        queue.push(layer);

    // Keep track of visited layers;
    std::set<Layer*> visited;

    /* Do breadth first search on the model and create streams */
    while (not queue.empty()) {
        auto from_layer = queue.front();
        queue.pop();
        visited.insert(from_layer);
#ifdef PARALLEL
        streams.push_back(new Stream(from_layer, state, state->state_stream));
#else
        streams.push_back(new Stream(from_layer, state));
#endif

        for (auto conn : from_layer->get_output_connections())
            if (visited.find(conn->to_layer) == visited.end())
                queue.push(conn->to_layer);
    }
}

SequentialStreamCluster::~SequentialStreamCluster() {
    // Delete streams
    for (auto stream : streams) delete stream;
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void SequentialStreamCluster::launch_calculations() {
#ifdef PARALLEL
    // Ensure all layer streams wait for clear event
    // Ensure all layer streams wait for input event
    // This function should not be called until this event has been
    //   scheduled to be recorded
    wait_event(this->state->clear_event);
    wait_event(this->state->input_event);
#endif
    for (auto& stream : streams) {
        for (auto& inst : stream->get_instructions())
            inst->activate();
        state->update_states(stream->to_layer);
    }
}

void SequentialStreamCluster::launch_weight_update() {
#ifdef PARALLEL
    wait_event(this->state->state_event);
#endif
    for (auto& stream : streams)
        for (auto& inst : stream->get_instructions())
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
