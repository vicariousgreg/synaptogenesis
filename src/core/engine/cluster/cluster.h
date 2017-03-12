#ifndef cluster_h
#define cluster_h

#include <map>
#include <string>

#include "model/structure.h"
#include "engine/cluster/cluster_node.h"
#include "engine/instruction.h"
#include "io/environment.h"
#include "util/constants.h"
#include "util/resource_manager.h"

class Cluster {
    public:
        Cluster(State *state, Environment *environment)
                : state(state),
                  environment(environment) {
            auto res_man = ResourceManager::get_instance();
            for (DeviceID i = 0 ; i < res_man->get_num_devices(); ++i)
                io_streams.push_back(res_man->create_stream(i));
        }
        virtual ~Cluster() { for (auto stream : io_streams) delete stream; }

        virtual void launch_pre_input_calculations() { };
        virtual void launch_input() = 0;
        virtual void launch_post_input_calculations() = 0;
        virtual void launch_state_update() { }
        virtual void launch_weight_update() = 0;
        virtual void launch_output() = 0;

        void wait_for_input() {
            for (auto& node : nodes)
                node->synchronize_input();
        }

        void wait_for_output() {
            for (auto& node : nodes)
                node->synchronize_output();
        }

        const std::vector<ClusterNode*> get_nodes() { return nodes; }

    protected:
        State *state;
        Environment *environment;
        std::vector<Stream*> io_streams;
        std::vector<ClusterNode*> nodes;
};

typedef std::vector<IOType> IOTypeVector;

class ParallelCluster : public Cluster {
    public:
        ParallelCluster(Structure *structure, State *state,
            Environment *environment);
        virtual ~ParallelCluster();

        virtual void launch_pre_input_calculations();
        virtual void launch_input();
        virtual void launch_post_input_calculations();
        virtual void launch_state_update();
        virtual void launch_weight_update();
        virtual void launch_output();

    protected:
        InstructionList sort_instructions(IOTypeVector types);

        std::vector<ClusterNode*> sorted_nodes[sizeof(IOTypes)];

        InstructionList pre_input_instructions;
        InstructionList post_input_instructions;
};

class SequentialCluster : public Cluster {
    public:
        SequentialCluster(Structure *structure, State *state,
            Environment *environment);
        virtual ~SequentialCluster();

        virtual void launch_input();
        virtual void launch_post_input_calculations();
        virtual void launch_output();
        virtual void launch_weight_update();

        std::vector<Stream*> compute_streams;
};

class FeedforwardCluster : public SequentialCluster {
    public:
        FeedforwardCluster(Structure *structure, State *state,
            Environment *environment);

        virtual void launch_weight_update();
};

inline Cluster *build_cluster(Structure *structure,
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

#endif
