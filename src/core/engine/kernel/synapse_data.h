#ifndef synapse_data_h
#define synapse_data_h

#include "model/connection.h"
#include "model/dendritic_node.h"
#include "engine/kernel/extractor.h"
#include "util/parallel.h"
#include "util/pointer.h"

class State;
class Attributes;

/* Data package that is passed into synaptic kernel functions */
class SynapseData {
    public:
        SynapseData(DendriticNode *parent_node, Connection *conn, State *state);

        /* Attributes pointer */
        const Attributes *from_attributes;
        const Attributes *to_attributes;

        /* Output extractor */
        const EXTRACTOR extractor;

        /* Connection attributes */
        Opcode opcode;
        bool convolutional;
        int row_stride, column_stride;
        int row_field_size, column_field_size;
        int row_offset, column_offset;
        int delay;
        bool second_order;

        /* Weight attributes */
        Pointer<float> weights;
        int num_weights;
        bool plastic;
        float max_weight;

        /* Layer attributes */
        int from_size, from_rows, from_columns;
        int to_size, to_rows, to_columns;
        int from_start_index, to_start_index;

        /* IO attributes */
        OutputType output_type;
        Pointer<Output> outputs;
        Pointer<Output> destination_outputs;
        Pointer<float> inputs;
        Pointer<float> second_order_inputs;
};

#endif
