#include "engine/engine.h"

void Engine::stage_clear() {
    // Reset stream cluster and state for timestep
    state->reset();
    stream_cluster.launch_non_input_calculations();
}

void Engine::stage_input() {
    // Start IO streaming
    state->transfer_input();

#ifdef PARALLEL
    stream_cluster.launch_input_calculations();
    if (learning_flag) stream_cluster.launch_weight_update();

    // Wait for IO to finish
    state->wait_for_input();
#endif
}

void Engine::stage_output() {
    // Start IO streaming
    state->transfer_output();

#ifdef PARALLEL
    state->update_all_states();

    // Wait for IO to finish
    state->wait_for_output();
#endif
}

void Engine::stage_calc() {
#ifdef PARALLEL
    // Synchronize and check for errors
    cudaSync();
    cudaCheckError(NULL);
#else
    stream_cluster.launch_input_calculations();
    if (learning_flag) stream_cluster.launch_weight_update();
    state->update_all_states();
#endif
}
