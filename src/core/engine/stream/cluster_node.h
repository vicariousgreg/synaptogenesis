#ifndef cluster_node_h
#define cluster_node_h

#include "model/layer.h"
#include "io/environment.h"
#include "engine/stream/instruction.h"
#include "util/parallel.h"

class ClusterNode {
    public:
        ClusterNode(Layer *layer, State *state, Environment *environment,
            Stream *io_stream, Stream *compute_stream);
        virtual ~ClusterNode();

        void activate_output_instruction();
        void activate_input_instruction();
        void activate_state_instruction();
        const InstructionList& get_instructions() const {
            return instructions;
        }

        Event *get_finished_event() const { return finished_event; }
        Event *get_input_event() const { return input_event; }
        Event *get_output_event() const { return output_event; }
        Event *get_state_event() const { return state_event; }
        Layer* const to_layer;

        Stream* const io_stream;
        Stream* const compute_stream;

    private:
        void dendrite_DFS(DendriticNode *curr);
        void set_input_instruction(Instruction *inst);
        void set_output_instruction(Instruction *inst);
        void set_state_instruction(Instruction *inst);
        void add_instruction(Instruction *inst);

        State* const state;
        Environment* const environment;
        InstructionList instructions;
        Instruction *input_instruction;
        Instruction *output_instruction;
        Instruction *state_instruction;

        Event *finished_event;
        Event *input_event;
        Event *output_event;
        Event *state_event;
};

#endif
