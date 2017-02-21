#include "engine/engine.h"

void Engine::stage_clear() {
    // Launch pre-input calculations
    stream_cluster->launch_pre_input_calculations();
}

void Engine::stage_output() {
    // Start output streaming
    stream_cluster->launch_output();

#ifdef PARALLEL
    // Wait for output
    stream_cluster->wait_for_output();
#endif
}

void Engine::stage_input() {
    stream_cluster->launch_input();

#ifdef PARALLEL
    stream_cluster->launch_post_input_calculations();
    stream_cluster->launch_state_update();
    if (learning_flag) stream_cluster->launch_weight_update();

    // Wait for input
    stream_cluster->wait_for_input();
#endif
}

void Engine::stage_calc() {
#ifdef PARALLEL
    // Synchronize and check for errors
    cudaSync();
    cudaCheckError(NULL);
#else
    stream_cluster->launch_post_input_calculations();
    stream_cluster->launch_state_update();
    if (learning_flag) stream_cluster->launch_weight_update();
#endif
}
