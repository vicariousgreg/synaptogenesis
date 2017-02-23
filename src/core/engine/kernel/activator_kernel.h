#ifndef activator_kernel_h
#define activator_kernel_h

#include "engine/kernel/synapse_kernel.h"

/* Activators are responsible for performing connection computation */
SYNAPSE_KERNEL get_base_activator_kernel(ConnectionType conn_type);
GLOBAL void activate_fully_connected(SynapseData synapse_data);
GLOBAL void activate_one_to_one(SynapseData synapse_data);
GLOBAL void activate_convergent(SynapseData synapse_data);

// Internal activator
GLOBAL void calc_internal(int size, Pointer<float> *src, Pointer<float> *dst, bool clear=false);

#define CALC_VAL(from_index, weight_index) \
    float val = extractor(outputs[from_index], delay) * weights[weight_index];

#define AGGREGATE(to_index, sum) \
    inputs[to_index] = calc(opcode, inputs[to_index], sum);

#define ACTIVATE_FULLY_CONNECTED(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_FULLY_CONNECTED(FUNC_NAME, \
    /* EXTRACTIONS */ \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE
     * Initialize sum to 0.0 */ \
    float sum = 0.0;, \
 \
    /* WEIGHT_OP
     * Calculate weight input, add to sum */ \
    CALC_VAL(from_index, weight_index); \
    sum += val; \
    UPDATE_CALC;, \
 \
    /* NEURON_POST
     * Aggregate sum to input */ \
    AGGREGATE(to_index, sum); \
)

#define ACTIVATE_ONE_TO_ONE(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_ONE_TO_ONE(FUNC_NAME, \
    /* EXTRACTIONS */ \
    UPDATE_EXT;, \
 \
    /* WEIGHT_OP
     * Calculate weight input, add to sum
     * Aggregate weight input to total input */ \
    CALC_VAL(index, index); \
    UPDATE_CALC; \
    AGGREGATE(index, val); \
)

#define ACTIVATE_CONVERGENT(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_CONVERGENT(FUNC_NAME, \
    /* EXTRACTIONS */ \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE
     * Initialize sum to 0.0 */ \
    float sum = 0.0;, \
 \
    /* WEIGHT_OP
     * Calculate weight input, add to sum */ \
    CALC_VAL(from_index, weight_index); \
    sum += val; \
    UPDATE_CALC;, \
 \
    /* NEURON_POST
     * Aggregate sum to input */ \
    AGGREGATE(to_index, sum); \
)

#endif
