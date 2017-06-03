#include "engine/engine.h"
#include "engine/instruction.h"
#include "engine/cluster/cluster.h"
#include "model/model.h"
#include "state/state.h"
#include "io/environment.h"

Engine::Engine(State *state, Environment *environment)
        : state(state),
          environment(environment),
          learning_flag(true) {
    // Create the clusters and gather their nodes
    for (auto& structure : state->model->get_structures()) {
        auto cluster = build_cluster(
            structure, state, environment);
        clusters.push_back(cluster);
        for (auto& node : cluster->get_nodes())
            cluster_nodes[node->to_layer] = node;
    }

    // Add external dependencies to the nodes
    for (auto& cluster : clusters)
        cluster->add_external_dependencies(cluster_nodes);
}

Engine::~Engine() {
    for (auto& cluster : clusters)
        delete cluster;
    InterDeviceInstruction::get_originals()->clear();
}

void Engine::stage_clear() {
    // Launch inter-device transfers
    for (auto inst : *InterDeviceInstruction::get_originals())
        inst->transfer();

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
