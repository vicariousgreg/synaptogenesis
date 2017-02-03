#ifndef kernel_data_h
#define kernel_data_h

#include "model/connection.h"
#include "util/parallel.h"

class State;
class Attributes;

/* Data package that is passed into kernel functions */
class KernelData {
    public:
        KernelData(Connection *conn, State *state);

        /* Neuron attributes */
        const Attributes *attributes;

        /* Connection attributes */
        Opcode opcode;
        bool convolutional;
        int field_size, stride;
        int fray;
        int delay;

        /* Weight attributes */
        float *weights;
        int num_weights;
        bool plastic;
        float max_weight;

        /* Layer attributes */
        int from_size, from_rows, from_columns;
        int to_size, to_rows, to_columns;

        /* IO attributes */
        OutputType output_type;
        Output *outputs;
        float *inputs;
};

#endif
