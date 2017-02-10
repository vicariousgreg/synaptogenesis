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

        const InstructionList& get_instructions() const { return instructions; }

#ifdef PARALLEL
        cudaStream_t get_cuda_stream() const { return cuda_stream; }
        cudaEvent_t* get_finished_event() const { return finished_event; }
#endif
        Layer* const to_layer;

    private:
        void dendrite_DFS(DendriticNode *curr);
        void add_instruction(Instruction *inst);

        State* const state;
        InstructionList instructions;

#ifdef PARALLEL
        cudaEvent_t *finished_event;
        cudaStream_t cuda_stream;
        bool external_stream;
#endif
};

#endif
