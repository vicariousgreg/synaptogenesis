#ifndef synapse_data_h
#define synapse_data_h

#include "network/connection.h"
#include "network/layer.h"
#include "network/dendritic_node.h"
#include "state/weight_matrix.h"
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
        const SubsetConfig subset_config;
        const ArborizedConfig arborized_config;
        const Connection connection;
        int connection_index;

        /* Weight attributes */
        const WeightMatrix* matrix;
        Pointer<float> weights;
        Pointer<float> second_order_weights;

        /* Layer attributes */
        int to_layer_index;
        int to_start_index;
        const Layer from_layer;
        const Layer to_layer;

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
