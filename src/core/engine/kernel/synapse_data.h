#ifndef synapse_data_h
#define synapse_data_h

#include "network/layer.h"
#include "state/weight_matrix.h"
#include "engine/kernel/extractor.h"
#include "engine/kernel/aggregator.h"
#include "util/parallel.h"
#include "util/pointer.h"

class State;
class Attributes;
class DendriticNode;
class Connection;

/* Data package that is passed into synaptic kernel functions */
class SynapseData {
    public:
        SynapseData(DendriticNode *parent_node, Connection *conn, State *state);

        /* Attributes pointer for to_layer */
        const Attributes * const attributes;

        /* Output extractor */
        const EXTRACTOR extractor;

        /* Input sum aggregator based on opcode */
        const AGGREGATOR aggregator;

        /* Connection attributes */
        const SubsetConfig subset_config;
        const ArborizedConfig arborized_config;
        const Connection connection;

        /* Weight attributes */
        const WeightMatrix * const matrix;
        const WeightMatrix * const second_order_host_matrix;

        /* Layer attributes */
        const Layer from_layer;
        const Layer to_layer;

        /* IO attributes */
        Pointer<Output> outputs;
        Pointer<Output> destination_outputs;
        Pointer<float> inputs;
};

/* Typedef for kernel functions, which just take SynapseData */
typedef SynapseData SYNAPSE_ARGS;
typedef void(*SYNAPSE_KERNEL)(SYNAPSE_ARGS);

#endif
