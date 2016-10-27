#ifndef instruction_h
#define instruction_h

#include "model/connection.h"
#include "state/state.h"
#include "driver/kernel.h"
#include "util/parallel.h"

class Instruction {
    public:
        Instruction(Connection *conn, State *state);

#ifdef PARALLEL
        void execute(cudaStream_t *stream);
        dim3 blocks_per_grid;
        dim3 threads_per_block;
#else
        void execute();
#endif

        ConnectionType type;
        bool convolutional;
        Opcode opcode;
        KERNEL kernel;
        EXTRACTOR extractor;

        int overlap, stride;
        int fray;
        int delay;

        int from_size, from_rows, from_columns;
        int to_size, to_rows, to_columns;
        int num_weights;

        OutputType output_type;
        Output *outputs;
        float *inputs;
        float *weights;
};

#endif
