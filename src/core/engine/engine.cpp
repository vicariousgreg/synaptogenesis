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

    // Process inter-device instructions
    for (auto& cluster : clusters) {
        for (auto& node : cluster->get_nodes()) {
            for (auto& pair : node->get_synapse_instructions()) {
                auto conn = pair.first;
                auto syn_inst = pair.second;

                // If inter-device, find or create corresponding transfer instruction
                if (state->is_inter_device(conn)) {
                    InterDeviceTransferInstruction *inst = nullptr;

                    // Search for existing instruction
                    for (auto inter_inst : this->inter_device_transfers) {
                        if (inter_inst->matches(conn, state)) {
                            inst = inter_inst;
                            break;
                        }
                    }

                    // Create if doesn't exist
                    // Add synapse instruction's dependencies
                    if (inst == nullptr) {
                        inst = new InterDeviceTransferInstruction(conn, state);
                        this->inter_device_transfers.push_back(inst);
                        inst->copy_dependencies(syn_inst);
                    }

                    // Set the synapse instruction to depend on transfer
                    syn_inst->add_dependency(inst);
                }
            }
        }
    }
}

Engine::~Engine() {
    for (auto& cluster : clusters)
        delete cluster;
    for (auto& inst : inter_device_transfers) delete inst;
}

void Engine::stage_clear() {
    // Launch inter-device transfers
    for (auto inst : this->inter_device_transfers)
        inst->activate();

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
