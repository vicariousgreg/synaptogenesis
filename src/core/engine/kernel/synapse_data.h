#ifndef synapse_data_h
#define synapse_data_h

#include "model/connection.h"
#include "engine/kernel/extractor.h"
#include "util/parallel.h"
#include "util/pointer.h"

class State;
class Attributes;

/* Data package that is passed into synaptic kernel functions */
class SynapseData {
    public:
        SynapseData(Connection *conn, State *state);

        /* Output extractor */
        const EXTRACTOR extractor;

        /* Connection attributes */
        Opcode opcode;
        bool convolutional;
        int field_size, stride;
        int fray;
        int delay;

        /* Weight attributes */
        Pointer<float> weights;
        int num_weights;
        bool plastic;
        float max_weight;

        /* Layer attributes */
        int from_size, from_rows, from_columns;
        int to_size, to_rows, to_columns;

        /* IO attributes */
        OutputType output_type;
        Pointer<Output> outputs;
        Pointer<Output> destination_outputs;
        Pointer<float> inputs;
};

#endif
