#include "engine/engine.h"

void Engine::stage_clear() {
    // Launch pre-input calculations
    for (auto& cluster : clusters)
        cluster->launch_pre_input_calculations();
}

void Engine::stage_input() {
    // Launch input transfer
    for (auto& cluster : clusters)
        cluster->launch_input();

    // Wait for input
    for (auto& cluster : clusters)
        cluster->wait_for_input();
}

void Engine::stage_calc() {
    for (auto& cluster : clusters) {
        // Launch post-input calculations
        cluster->launch_post_input_calculations();

        // Launch state update
        cluster->launch_state_update();

        // Launch weight updates
        if (learning_flag) cluster->launch_weight_update();
    }
}

void Engine::stage_output() {
    // Start output streaming
    for (auto& cluster : clusters)
        cluster->launch_output();

    // Wait for output
    for (auto& cluster : clusters)
        cluster->wait_for_output();
}
