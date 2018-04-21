#include "engine/cluster/cluster.h"
#include "engine/cluster/cluster_node.h"
#include "engine/instruction.h"
#include "network/structure.h"
#include "state/state.h"
#include "engine/engine.h"
#include "util/resources/resource_manager.h"

Cluster::Cluster(State *state, Engine *engine, PropertyConfig args)
        : state(state),
          engine(engine) {
    auto active_devices = state->get_active_devices();
    auto res_man = ResourceManager::get_instance();
    for (auto device_id : active_devices)
        io_streams[device_id] = res_man->create_stream(device_id);
}

Cluster::~Cluster() {
    for (auto& node : nodes) delete node;
}

void Cluster::add_external_dependencies(
        std::map<Layer*, ClusterNode*> all_nodes) {
    // Crawl through the nodes and add dependencies for state updates
    // This prevents race conditions from output updates
    // Ensure that the output is not updated until it's been transferred
    for (auto& node : nodes) {
        for (auto& syn_inst : node->get_synapse_activate_instructions()) {
            auto conn = syn_inst->connection;
            all_nodes[conn->from_layer]
                ->get_state_update_instruction()->add_dependency(syn_inst);
            syn_inst->add_dependency(
                all_nodes[conn->from_layer]->get_state_update_instruction());
        }
    }
}

void Cluster::launch_input() {
    for (auto& node : nodes)
        node->activate_input();
}

void Cluster::launch_output() {
    for (auto& node : nodes)
        node->activate_output();
}

void Cluster::wait_for_input() {
    for (auto& node : nodes)
        node->synchronize_input();
}

void Cluster::wait_for_output() {
    for (auto& node : nodes)
        node->synchronize_output();
}

Cluster *build_cluster(Structure *structure,
        State *state, Engine *engine, PropertyConfig args) {
    switch (structure->cluster_type) {
        case PARALLEL:
            return new ParallelCluster(structure, state, engine, args);
        case SEQUENTIAL:
            return new SequentialCluster(structure, state, engine, args);
        case FEEDFORWARD:
            return new FeedforwardCluster(structure, state, engine, args);
        default:
            LOG_ERROR(
                "Error building cluster for " + structure->str() + ":\n"
                "  Unrecognized stream cluster type!");
    }
    return nullptr;
}
