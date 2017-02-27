#include "engine/engine.h"

void Engine::stage_clear() {
    // Launch pre-input calculations
    for (auto& cluster : clusters)
        cluster->launch_pre_input_calculations();
}

void Engine::stage_output() {
    // Start output streaming
    for (auto& cluster : clusters)
        cluster->launch_output();

#ifdef __CUDACC__
    // Wait for output
    for (auto& cluster : clusters)
        cluster->wait_for_output();
#endif
}

void Engine::stage_input() {
    for (auto& cluster : clusters)
        cluster->launch_input();

#ifdef __CUDACC__
    for (auto& cluster : clusters) {
        cluster->launch_post_input_calculations();
        cluster->launch_state_update();
        if (learning_flag) cluster->launch_weight_update();

        // Wait for input
        cluster->wait_for_input();
    }
#endif
}

void Engine::stage_calc() {
    // Synchronize and check for errors
    device_synchronize();
    device_check_error(NULL);

#ifdef __CUDACC__
#else
    for (auto& cluster : clusters) {
        cluster->launch_post_input_calculations();
        cluster->launch_state_update();
        if (learning_flag) cluster->launch_weight_update();
    }
#endif
}
