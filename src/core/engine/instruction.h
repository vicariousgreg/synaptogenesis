#ifndef instruction_h
#define instruction_h

#include "model/connection.h"
#include "state/state.h"
#include "engine/kernel.h"
#include "util/parallel.h"

class ConnectionData {
    public:
        ConnectionData(Connection *conn, State *state);

        bool convolutional;
        Opcode opcode;

        EXTRACTOR extractor;

        int overlap, stride;
        int fray;
        int delay;

        int from_size, from_rows, from_columns;
        int to_size, to_rows, to_columns;
        int num_weights;
        bool plastic;
        float max_weight;

        OutputType output_type;
        Output *outputs;
        float *inputs;
        float *weights;
};

class Instruction {
    public:
        Instruction(Connection *conn, State *state);

        bool is_plastic() { return this->connection_data.plastic; }
        void disable_learning();

#ifdef PARALLEL
        void execute(cudaStream_t *stream);
        void update(cudaStream_t *stream);
        dim3 blocks_per_grid;
        dim3 threads_per_block;
#else
        void execute();
        void update();
#endif

        ConnectionType type;

        EXTRACTOR extractor;
        ACTIVATOR activator;
        UPDATER updater;

        ConnectionData connection_data;
};

#endif
