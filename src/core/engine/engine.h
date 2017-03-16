#ifndef engine_h
#define engine_h

#include "state/state.h"
#include "model/model.h"
#include "io/environment.h"
#include "engine/cluster/cluster.h"

class Engine {
    public:
        Engine(State *state, Environment *environment)
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

        virtual ~Engine() {
            for (auto& cluster : clusters)
                delete cluster;
        }

        void set_learning_flag(bool status) { learning_flag = status; }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_output();
        void stage_calc();

    protected:
        State *state;
        Environment *environment;
        std::vector<Cluster*> clusters;
        std::map<Layer*, ClusterNode*> cluster_nodes;
        bool learning_flag;
};

#endif
