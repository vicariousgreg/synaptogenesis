#ifndef stream_h
#define stream_h

#include "model/layer.h"
#include "engine/instruction.h"
#include "util/parallel.h"

class Stream {
    public:
        Stream(Layer *layer, State *state);
#ifdef PARALLEL
        Stream(Layer *layer, State *state, cudaStream_t cuda_stream);
#endif
        virtual ~Stream();

        void activate_output_instruction();
        void activate_input_instruction();
        void activate_state_instruction();
        const InstructionList& get_instructions() const {
            return instructions;
        }

#ifdef PARALLEL
        cudaStream_t get_cuda_stream() const { return cuda_stream; }
        cudaEvent_t get_finished_event() const { return finished_event; }
        cudaEvent_t get_input_event() const { return input_event; }
        cudaEvent_t get_output_event() const { return output_event; }
        cudaEvent_t get_state_event() const { return state_event; }
#endif
        Layer* const to_layer;

    private:
        void init();
        void dendrite_DFS(DendriticNode *curr);
        void set_input_instruction(Instruction *inst);
        void set_output_instruction(Instruction *inst);
        void set_state_instruction(Instruction *inst);
        void add_instruction(Instruction *inst);

        State* const state;
        InstructionList instructions;
        Instruction *input_instruction;
        Instruction *output_instruction;
        Instruction *state_instruction;

#ifdef PARALLEL
        cudaEvent_t finished_event;
        cudaEvent_t input_event;
        cudaEvent_t output_event;
        cudaEvent_t state_event;
        cudaStream_t cuda_stream;
        bool external_stream;
#endif
};

#endif
