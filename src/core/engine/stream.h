#ifndef stream_h
#define stream_h

#include "model/layer.h"
#include "io/environment.h"
#include "engine/instruction.h"
#include "util/parallel.h"

class Stream {
    public:
        Stream(Layer *layer, State *state, Environment *environment);
#ifdef __CUDACC__
        Stream(Layer *layer, State *state, Environment *environment, cudaStream_t cuda_stream);
#endif
        virtual ~Stream();

        void activate_output_instruction();
        void activate_input_instruction();
        void activate_state_instruction();
        const InstructionList& get_instructions() const {
            return instructions;
        }

#ifdef __CUDACC__
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
        Environment* const environment;
        InstructionList instructions;
        Instruction *input_instruction;
        Instruction *output_instruction;
        Instruction *state_instruction;

#ifdef __CUDACC__
        cudaEvent_t finished_event;
        cudaEvent_t input_event;
        cudaEvent_t output_event;
        cudaEvent_t state_event;
        cudaStream_t cuda_stream;
        bool external_stream;
#endif
};

#endif
