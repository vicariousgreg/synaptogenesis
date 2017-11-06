#ifndef cluster_h
#define cluster_h

#include <map>
#include <vector>

#include "engine/cluster/cluster_node.h"
#include "util/constants.h"
#include "util/property_config.h"
#include "util/stream.h"

class Layer;
class Structure;
class State;
class Engine;
class Instruction;
typedef std::vector<Instruction*> InstructionList;

class Cluster {
    public:
        Cluster(State *state, Engine *engine, PropertyConfig args);
        virtual ~Cluster();

        void add_external_dependencies(
            std::map<Layer*, ClusterNode*> all_nodes);
        virtual void add_inter_device_instruction(
            Instruction *synapse_instruction,
            Instruction *inter_device_instruction,
            bool new_transfer) = 0;

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
        Engine *engine;
        std::map<DeviceID, Stream*> io_streams;
        std::vector<ClusterNode*> nodes;

};

class ParallelCluster : public Cluster {
    public:
        ParallelCluster(Structure *structure, State *state,
            Engine *engine, PropertyConfig args);

        virtual void add_inter_device_instruction(
            Instruction *synapse_instruction,
            Instruction *inter_device_instruction,
            bool new_transfer);

        virtual void launch_pre_input_calculations();
        virtual void launch_post_input_calculations();
        virtual void launch_state_update();
        virtual void launch_weight_update();

    protected:
        InstructionList sort_instructions(Engine *engine,
            IOTypeMask include, IOTypeMask exclude, bool plastic);

        InstructionList inter_device_instructions;
        InstructionList pre_input_instructions;
        InstructionList post_input_instructions;
        InstructionList plastic_instructions;
};

class SequentialCluster : public Cluster {
    public:
        SequentialCluster(Structure *structure, State *state,
            Engine *engine, PropertyConfig args);

        virtual void add_inter_device_instruction(
            Instruction *synapse_instruction,
            Instruction *inter_device_instruction,
            bool new_transfer);

        virtual void launch_post_input_calculations();
        virtual void launch_weight_update();

        std::map<DeviceID, Stream*> compute_streams;
};

class FeedforwardCluster : public SequentialCluster {
    public:
        FeedforwardCluster(Structure *structure, State *state,
            Engine *engine, PropertyConfig args);
};

Cluster *build_cluster(Structure *structure,
        State *state, Engine *engine, PropertyConfig args=PropertyConfig());

#endif
