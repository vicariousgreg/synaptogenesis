#include "engine/stream_cluster.h"

StreamCluster::StreamCluster(Model *model, State *state)
        : state(state) {
    // Build instructions
    for (auto& layer : model->get_layers())
        // Skip layers with no input connections (typically sensory)
        if (layer->get_input_connections().size() > 0) 
            streams[layer->get_type()].push_back(new Stream(layer, state));

    // Schedule input instructions
    schedule(INPUT);
    schedule(INPUT_OUTPUT);
    this->sort_schedule(this->input_instructions);

    // Schedule non input instructions
    schedule(OUTPUT);
    schedule(INTERNAL);
    this->sort_schedule(this->non_input_instructions);

    // Schedule plastic
    schedule_plastic();
    this->sort_schedule(this->plastic_instructions);
}

StreamCluster::~StreamCluster() {
    // Delete streams
    for (auto type : IOTypes)
        for (auto stream : streams[type])
            delete stream;
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void StreamCluster::launch_non_input_calculations() {
#ifdef PARALLEL
    // Ensure all layer streams wait for clear event
    wait_event(OUTPUT, this->state->clear_event);
    wait_event(INTERNAL, this->state->clear_event);
#endif

    // Launch clear output relevant computations
    for (auto& inst : this->non_input_instructions) inst->activate();
}

void StreamCluster::launch_input_calculations() {
#ifdef PARALLEL
    // Ensure all layer streams wait for input event
    // This function should not be called until this event has been
    //   scheduled to be recorded
    wait_event(INPUT, this->state->input_event);
    wait_event(INPUT_OUTPUT, this->state->input_event);
#endif

    // Launch input output relevant computations
    for (auto& inst : this->input_instructions) inst->activate();

#ifdef PARALLEL
    block_stream_to(INPUT, state->state_stream);
    block_stream_to(INPUT_OUTPUT, state->state_stream);
    block_stream_to(OUTPUT, state->state_stream);
    block_stream_to(INTERNAL, state->state_stream);
#endif
}

void StreamCluster::launch_weight_update() {
#ifdef PARALLEL
    wait_event(INPUT, this->state->state_event);
    wait_event(INPUT_OUTPUT, this->state->state_event);
    wait_event(OUTPUT, this->state->state_event);
    wait_event(INTERNAL, this->state->state_event);
#endif
    for (auto& inst : this->plastic_instructions) inst->update();
}

/******************************************************************************/
/***************************** SCHEDULERS *************************************/
/******************************************************************************/

void StreamCluster::schedule(IOType to_type) {
    for (auto& stream : streams[to_type])
        for (auto& inst : stream->get_instructions())
            schedules[stream->to_layer].push_back(inst);
}

void StreamCluster::schedule_plastic() {
    for (auto type : IOTypes)
        for (auto& stream : streams[type])
            for (auto& inst : stream->get_instructions())
                if (inst->is_plastic())
                    schedules[stream->to_layer].push_back(inst);
}

void StreamCluster::sort_schedule(InstructionList &destination) {
#ifdef PARALLEL
    // Perform round robin on streams
    // Connections should be initialized this way to take advantage of
    //   stream overlap
    bool done = false;
    for (int i = 0; not done; ++i) {
        done = true;
        for (auto& schedule : this->schedules) {
            if (i < schedule.second.size()) {
                done = false;
                destination.push_back(schedule.second[i]);
            }
        }
    }
#else
    // Copy over to schedule
    for (auto& schedule : this->schedules)
        for (int i = 0; i < schedule.second.size(); ++i)
            destination.push_back(schedule.second[i]);
#endif

    // Clear schedule
    for (auto& schedule : this->schedules)
        schedule.second.clear();
}

/******************************************************************************/
/**************************** EVENT HANDLING **********************************/
/******************************************************************************/

#ifdef PARALLEL
void StreamCluster::wait_event(IOType to_type, cudaEvent_t *event) {
    for (auto& stream : streams[to_type])
        cudaStreamWaitEvent(stream->get_cuda_stream(), *event, 0);
}

void StreamCluster::block_stream_to(IOType to_type, cudaStream_t cuda_stream) {
    for (auto& stream : streams[to_type])
        cudaStreamWaitEvent(cuda_stream, *stream->get_finished_event(), 0);
}
#endif
