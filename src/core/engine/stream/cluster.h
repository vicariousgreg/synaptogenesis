#ifndef cluster_h
#define cluster_h

#include <map>
#include <string>

#include "model/structure.h"
#include "engine/stream/cluster_node.h"
#include "engine/stream/instruction.h"
#include "io/environment.h"
#include "util/parallel.h"
#include "util/constants.h"

class Cluster {
    public:
        Cluster(State *state, Environment *environment)
                : state(state),
                  environment(environment) { }
        virtual ~Cluster() { }

        virtual void launch_pre_input_calculations() { };
        virtual void launch_input() = 0;
        virtual void launch_post_input_calculations() = 0;
        virtual void launch_output() = 0;
        virtual void launch_state_update() { }
        virtual void launch_weight_update() = 0;

        virtual void wait_for_input() = 0;
        virtual void wait_for_output() = 0;

    protected:
        State *state;
        Environment *environment;
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
        virtual void launch_output();
        virtual void launch_state_update();
        virtual void launch_weight_update();

        virtual void wait_for_input();
        virtual void wait_for_output();

    protected:
        InstructionList sort_instructions(IOTypeVector types, bool plastic);

        std::vector<ClusterNode*> nodes[sizeof(IOTypes)];

        InstructionList pre_input_instructions;
        InstructionList post_input_instructions;
        InstructionList plastic_instructions;
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

        virtual void wait_for_input();
        virtual void wait_for_output();

        Stream *compute_stream;

    protected:
        std::vector<ClusterNode*> nodes;
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
    return NULL;
}

#endif
