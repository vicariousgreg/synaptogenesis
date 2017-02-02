#include "engine/engine.h"

void Engine::disable_learning() {
    stream_cluster.disable_learning();
}

void Engine::stage_clear() {
    // Reset stream cluster and state for timestep
    state->reset();
    stream_cluster.launch_clear_output_calculations();
}

void Engine::stage_input() {
    // Start input streaming
    state->transfer_input();

#ifdef PARALLEL
    // If parallel, schedule everything now
    // Events will ensure processing waits until ready

    // Launch output relevant computations
    stream_cluster.launch_input_output_calculations();

    // Output state computation
    state->update_output_states();

    // Launch remaining calculations
    stream_cluster.launch_non_output_calculations();

    // Launch final state computations
    state->update_non_output_states();

    // Schedule and launch weight updates
    stream_cluster.launch_weight_update();

    // Wait for input to finish
    state->wait_for_input();
#endif
}

void Engine::stage_calc_output() {
#ifdef PARALLEL
#else
    stream_cluster.launch_input_output_calculations();
    state->update_output_states();
#endif
}

void Engine::stage_send_output() {
    // Stream output
    state->transfer_output();

#ifdef PARALLEL
    // Wait for it to finish
    state->wait_for_output();
#endif
}

void Engine::stage_remaining() {
#ifdef PARALLEL
#else
    stream_cluster.launch_non_output_calculations();
    state->update_non_output_states();
#endif
}

void Engine::stage_weights() {
#ifdef PARALLEL
    // Synchronize and check for errors
    cudaSync();
    cudaCheckError(NULL);
#else
    stream_cluster.launch_weight_update();
#endif
}
