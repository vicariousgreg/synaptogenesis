#ifndef cluster_node_h
#define cluster_node_h

#include "model/layer.h"
#include "io/environment.h"
#include "engine/stream/instruction.h"

class ClusterNode {
    public:
        ClusterNode(Layer *layer, State *state, Environment *environment,
            Stream *io_stream, Stream *compute_stream);
        virtual ~ClusterNode();

        void activate_input();
        void activate_state();
        void activate_output();

        const InstructionList& get_instructions() const {
            return instructions;
        }

        void synchronize_input();
        void synchronize_output();

        Layer* const to_layer;
        Stream* const io_stream;
        Stream* const compute_stream;

    private:
        void dendrite_DFS(DendriticNode *curr);
        void add_instruction(Instruction *inst);
        void set_input_instruction(Instruction *inst);
        void set_input_copy_instruction(Instruction *inst);
        void set_state_instruction(Instruction *inst);
        void set_output_copy_instruction(Instruction *inst);
        void set_output_instruction(Instruction *inst);

        State* const state;
        Environment* const environment;

        InstructionList instructions;
        Instruction *state_instruction;

        bool is_input;
        Instruction *input_instruction;
        Instruction *input_copy_instruction;

        bool is_output;
        Instruction *output_instruction;
        Instruction *output_copy_instruction;

        Event *input_event;
        Event *input_copy_event;
        Event *output_copy_event;
        Event *output_event;
};

#endif
