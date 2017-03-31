#ifndef cluster_h
#define cluster_h

#include <map>
#include <vector>

#include "engine/cluster/cluster_node.h"
#include "util/constants.h"
#include "util/stream.h"

class Layer;
class Structure;
class State;
class Environment;
class Instruction;
typedef std::vector<Instruction*> InstructionList;

class Cluster {
    public:
        Cluster(State *state, Environment *environment);
        virtual ~Cluster();

        virtual void add_external_dependencies(
            std::map<Layer*, ClusterNode*> all_nodes) = 0;

        void launch_input();
        void launch_output();

        virtual void launch_pre_input_calculations() { };
        virtual void launch_post_input_calculations() = 0;
        virtual void launch_state_update() { }
        virtual void launch_weight_update() = 0;

        void wait_for_input();
        void wait_for_output();

        const std::vector<ClusterNode*> get_nodes() { return nodes; }

    protected:
        State *state;
        Environment *environment;
        std::vector<Stream*> io_streams;
        std::vector<ClusterNode*> nodes;
};

class ParallelCluster : public Cluster {
    public:
        ParallelCluster(Structure *structure, State *state,
            Environment *environment);

        virtual void add_external_dependencies(
            std::map<Layer*, ClusterNode*> all_nodes);

        virtual void launch_pre_input_calculations();
        virtual void launch_post_input_calculations();
        virtual void launch_state_update();
        virtual void launch_weight_update();

    protected:
        InstructionList sort_instructions(
            IOTypeMask include, IOTypeMask exclude, bool plastic);

        InstructionList pre_input_instructions;
        InstructionList post_input_instructions;
        InstructionList plastic_instructions;
};

class SequentialCluster : public Cluster {
    public:
        SequentialCluster(Structure *structure, State *state,
            Environment *environment);

        virtual void add_external_dependencies(
            std::map<Layer*, ClusterNode*> all_nodes);

        virtual void launch_post_input_calculations();
        virtual void launch_weight_update();

        std::vector<Stream*> compute_streams;
};

class FeedforwardCluster : public SequentialCluster {
    public:
        FeedforwardCluster(Structure *structure, State *state,
            Environment *environment);

        virtual void launch_weight_update();
};

Cluster *build_cluster(Structure *structure,
        State *state, Environment *environment);

#endif
