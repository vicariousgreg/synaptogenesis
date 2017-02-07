#include "engine/engine.h"

void Engine::stage_clear() {
    // Reset stream cluster and state for timestep
    state->reset();
    stream_cluster.launch_non_input_calculations();
}

void Engine::stage_output() {
    // Start output streaming
    state->transfer_output();

#ifdef PARALLEL
    // Wait for output
    state->wait_for_output();
#endif
}

void Engine::stage_input() {
    // Start input streaming
    state->transfer_input();

#ifdef PARALLEL
    stream_cluster.launch_input_calculations();
    state->update_states();
    if (learning_flag) stream_cluster.launch_weight_update();

    // Wait for input
    state->wait_for_input();
#endif
}

void Engine::stage_calc() {
#ifdef PARALLEL
    // Synchronize and check for errors
    cudaSync();
    cudaCheckError(NULL);
#else
    stream_cluster.launch_input_calculations();
    state->update_states();
    if (learning_flag) stream_cluster.launch_weight_update();
#endif
}
