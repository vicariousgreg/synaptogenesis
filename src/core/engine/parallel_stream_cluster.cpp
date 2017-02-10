#include <queue>

#include "engine/stream_cluster.h"

ParallelStreamCluster::ParallelStreamCluster(Model *model, State *state)
        : StreamCluster(model, state) {
    // Build instructions
    for (auto& layer : model->get_layers())
        // Skip layers with no input connections (typically sensory)
        if (layer->get_input_connections().size() > 0) 
            streams[layer->get_type()].push_back(new Stream(layer, state));

    // Schedule instructions
    post_input_instructions = sort_instructions(
        IOTypeVector { INPUT, INPUT_OUTPUT },
        false);
    pre_input_instructions = sort_instructions(
        IOTypeVector { OUTPUT, INTERNAL },
        false);
    plastic_instructions = sort_instructions(
        IOTypeVector { INPUT, INPUT_OUTPUT, OUTPUT, INTERNAL },
        true);
}

ParallelStreamCluster::~ParallelStreamCluster() {
    // Delete streams
    for (auto type : IOTypes)
        for (auto stream : streams[type]) delete stream;
}

InstructionList ParallelStreamCluster::sort_instructions(
        IOTypeVector types, bool plastic) {
#ifdef PARALLEL
    std::map<Layer*, std::queue<Instruction* > > schedules;
#endif
    InstructionList destination;

    // Extract instructions
    for (auto type : types)
        for (auto& stream : streams[type])
            for (auto& inst : stream->get_instructions())
                if (not plastic or inst->is_plastic())
#ifndef PARALLEL    // Add directly to destination list
                    destination.push_back(inst);
#else               // Add to schedule map for round robin
                    schedules[stream->to_layer].push(inst);

    // Perform round robin on streams
    // Connections should be initialized this way to take advantage of
    //   stream overlap
    while (schedules.size() > 0) {
        for (auto& schedule : schedules) {
            destination.push_back(schedule.second.front());
            schedule.second.pop();
            if (schedule.second.size() == 0)
                schedules.erase(schedule.first);
        }
    }
#endif
    return destination;
}

/******************************************************************************/
/****************************** LAUNCHERS *************************************/
/******************************************************************************/

void ParallelStreamCluster::launch_pre_input_calculations() {
#ifdef PARALLEL
    // Ensure all layer streams wait for clear event
    wait_event(OUTPUT, this->state->clear_event);
    wait_event(INTERNAL, this->state->clear_event);
#endif
    for (auto& inst : this->pre_input_instructions) inst->activate();
}

void ParallelStreamCluster::launch_post_input_calculations() {
#ifdef PARALLEL
    // Ensure all layer streams wait for input event
    // This function should not be called until this event has been
    //   scheduled to be recorded
    wait_event(INPUT, this->state->input_event);
    wait_event(INPUT_OUTPUT, this->state->input_event);
#endif
    for (auto& inst : this->post_input_instructions) inst->activate();
#ifdef PARALLEL
    block_stream_to(INPUT, state->state_stream);
    block_stream_to(INPUT_OUTPUT, state->state_stream);
    block_stream_to(OUTPUT, state->state_stream);
    block_stream_to(INTERNAL, state->state_stream);
#endif
}

void ParallelStreamCluster::launch_weight_update() {
#ifdef PARALLEL
    wait_event(INPUT, this->state->state_event);
    wait_event(INPUT_OUTPUT, this->state->state_event);
    wait_event(OUTPUT, this->state->state_event);
    wait_event(INTERNAL, this->state->state_event);
#endif
    for (auto& inst : this->plastic_instructions) inst->update();
}

/******************************************************************************/
/**************************** EVENT HANDLING **********************************/
/******************************************************************************/

#ifdef PARALLEL
void ParallelStreamCluster::wait_event(IOType to_type, cudaEvent_t *event) {
    for (auto& stream : streams[to_type])
        cudaStreamWaitEvent(stream->get_cuda_stream(), *event, 0);
}

void ParallelStreamCluster::block_stream_to(IOType to_type, cudaStream_t cuda_stream) {
    for (auto& stream : streams[to_type])
        cudaStreamWaitEvent(cuda_stream, *stream->get_finished_event(), 0);
}
#endif
