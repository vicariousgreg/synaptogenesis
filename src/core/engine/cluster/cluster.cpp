#include "engine/cluster/cluster.h"
#include "model/structure.h"
#include "state/state.h"
#include "io/environment.h"
#include "engine/cluster/cluster_node.h"
#include "util/resource_manager.h"

Cluster::Cluster(State *state, Environment *environment)
        : state(state),
          environment(environment) {
    auto res_man = ResourceManager::get_instance();
    for (DeviceID i = 0 ; i < res_man->get_num_devices(); ++i)
        io_streams.push_back(res_man->create_stream(i));
}
Cluster::~Cluster() {
    for (auto& node : nodes) delete node;
}

void Cluster::launch_input() {
    for (auto& node : nodes) node->activate_input();
}

void Cluster::launch_output() {
    for (auto& node : nodes) node->activate_output();
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
        State *state, Environment *environment) {
    if (not state->check_compatibility(structure))
        ErrorManager::get_instance()->log_error(
            "Cluster compatibility conflict detected!");

    switch (structure->cluster_type) {
        case(PARALLEL):
            return new ParallelCluster(structure, state, environment);
        case(SEQUENTIAL):
            return new SequentialCluster(structure, state, environment);
        case(FEEDFORWARD):
            return new FeedforwardCluster(structure, state, environment);
        default:
            ErrorManager::get_instance()->log_error(
                "Unrecognized stream cluster type!");
    }
    return nullptr;
}
