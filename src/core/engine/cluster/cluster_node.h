#ifndef cluster_node_h
#define cluster_node_h

#include <map>
#include <vector>

#include "util/stream.h"

class Layer;
class DendriticNode;
class Connection;
class State;
class Environment;
class Instruction;
typedef std::vector<Instruction*> InstructionList;

class ClusterNode {
    public:
        ClusterNode(Layer *layer, State *state, Environment *environment,
            Stream *io_stream, Stream *compute_stream);
        virtual ~ClusterNode();

        void activate_input();
        void activate_state();
        void activate_output();

        void synchronize_input();
        void synchronize_output();

        const InstructionList get_activate_instructions() const;
        const InstructionList get_update_instructions() const;
        Instruction* get_input_instruction() const;
        Instruction* get_state_update_instruction() const;
        Instruction* get_state_learning_instruction() const;
        Instruction* get_output_instruction() const;
        const std::map<Connection*, Instruction*>
            get_synapse_instructions() const;
        const std::map<Connection*, Instruction*>
            get_external_transfer_instructions();

        Layer* const to_layer;
        const DeviceID device_id;
        Stream* const io_stream;
        Stream* const compute_stream;

        State* const state;
        Environment* const environment;

    private:
        void dendrite_DFS(DendriticNode *curr);

        std::map<Connection*, Instruction*> synapse_instructions;

        InstructionList activate_instructions;
        InstructionList update_instructions;
        Instruction *state_update_instruction;
        Instruction *state_learning_instruction;

        bool is_input;
        Instruction *input_instruction;

        bool is_expected;
        Instruction *expected_instruction;

        bool is_output;
        Instruction *output_instruction;
};

#endif
