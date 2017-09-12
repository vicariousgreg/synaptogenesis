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

        /* Attributes pointer for to_layer */
        const Attributes *attributes;

        /* Output extractor */
        const EXTRACTOR extractor;

        /* Connection attributes */
        int connection_index;
        Opcode opcode;
        bool convolutional;
        const SubsetConfig subset_config;
        const ArborizedConfig arborized_config;
        int delay;

        /* Weight attributes */
        Pointer<float> weights;
        Pointer<float> second_order_weights;
        int num_weights;
        bool plastic;
        float max_weight;

        /* Layer attributes */
        int to_layer_index;
        int to_start_index;
        int from_size, from_rows, from_columns;
        int to_size, to_rows, to_columns;

        /* IO attributes */
        OutputType output_type;
        Pointer<Output> outputs;
        Pointer<Output> destination_outputs;
        Pointer<float> inputs;
};

/* Typedef for kernel functions, which just take SynapseData */
typedef SynapseData SYNAPSE_ARGS;
typedef void(*SYNAPSE_KERNEL)(SYNAPSE_ARGS);

#endif
