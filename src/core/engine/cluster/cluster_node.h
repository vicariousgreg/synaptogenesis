#ifndef cluster_node_h
#define cluster_node_h

#include <map>
#include <vector>

#include "util/resources/stream.h"

class Layer;
class DendriticNode;
class Connection;
class State;
class Engine;
class Instruction;
class SynapseInstruction;
typedef std::vector<Instruction*> InstructionList;
typedef std::vector<SynapseInstruction*> SynapseInstructionList;

class ClusterNode {
    public:
        ClusterNode(Layer *layer, State *state, Engine *engine,
            Stream *io_stream, Stream *compute_stream);
        virtual ~ClusterNode();

        void activate_input();
        void activate_state();
        void activate_output();

        void synchronize_input();
        void synchronize_output();

        const InstructionList& get_activate_instructions() const;
        const InstructionList& get_update_instructions() const;
        Instruction* get_init_instruction() const;
        Instruction* get_input_instruction() const;
        Instruction* get_state_update_instruction() const;
        Instruction* get_state_learning_instruction() const;
        Instruction* get_output_instruction() const;
        const SynapseInstructionList& get_synapse_activate_instructions() const;
        const SynapseInstructionList& get_synapse_update_instructions() const;

        Layer* const to_layer;
        const DeviceID device_id;
        Stream* const io_stream;
        Stream* const compute_stream;

        State* const state;
        Engine* const engine;

    private:
        void dendrite_DFS(DendriticNode *curr);

        SynapseInstructionList synapse_activate_instructions;
        SynapseInstructionList synapse_update_instructions;

        InstructionList activate_instructions;
        InstructionList update_instructions;
        Instruction *state_update_instruction;
        Instruction *state_learning_instruction;

        Instruction *init_instruction;

        Instruction *input_instruction;
        std::vector<Instruction*> input_auxiliary_instructions;

        Instruction *output_instruction;
        std::vector<Instruction*> output_auxiliary_instructions;
};

#endif
