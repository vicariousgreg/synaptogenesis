#include <algorithm>

#include "engine/kernel/activator_kernel.h"
#include "state/attributes.h"
#include "util/error_manager.h"

#define MOD_RATE 0.05
#define MOD_DECAY 0.005
#define MOD_MAX 10.0


// Different minimum functions are used on the host and device
#ifdef PARALLEL
#define MIN min
#else
#define MIN std::fmin
#endif

#define EXTRACT_TRACE \
    float *trace = weights + (2*num_weights);

#define CALC_VAL(from_index, weight_index) \
    float val = extractor(outputs[from_index], delay) * weights[weight_index];

#define UPDATE_TRACE(weight_index) \
    if (plastic) { \
        float old_trace = trace[weight_index]; \
        float new_trace = old_trace + (MOD_RATE * val) - (MOD_DECAY * old_trace); \
        trace[weight_index] = MIN(MOD_MAX, new_trace);  \
    }

#define AGGREGATE(to_index, sum) \
    inputs[to_index] = calc(opcode, inputs[to_index], sum);

/******************************************************************************/
/********************** CONNECTION ACTIVATOR KERNELS **************************/
/******************************************************************************/

KERNEL get_activator_kernel(ConnectionType conn_type) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return activate_fully_connected;
        case ONE_TO_ONE:
            return activate_one_to_one;
        case CONVERGENT:
        case CONVOLUTIONAL:
            return activate_convergent;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

KERNEL get_activator_kernel_trace(ConnectionType conn_type) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return activate_fully_connected_trace;
        case ONE_TO_ONE:
            return activate_one_to_one_trace;
        case CONVERGENT:
        case CONVOLUTIONAL:
            return activate_convergent_trace;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

#define ACTIVATE_FULLY_CONNECTED(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_FULLY_CONNECTED(FUNC_NAME, \
    /* EXTRACTIONS
     * Pointer to activation trace */ \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE
     * Initialize sum to 0.0 */ \
    float sum = 0.0;, \
 \
    /* WEIGHT_OP
     * Calculate weight input, add to sum, update modifying substance */ \
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
    /* EXTRACTIONS
     * Pointer to activation trace */ \
    UPDATE_EXT;, \
 \
    /* WEIGHT_OP
     * Calculate weight input, add to sum, update modifying substance
     * Aggregate weight input to total input */ \
    CALC_VAL(index, index); \
    UPDATE_CALC; \
    AGGREGATE(index, val); \
)

#define ACTIVATE_CONVERGENT(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_CONVERGENT(FUNC_NAME, \
    /* EXTRACTIONS
     * Pointer to activation trace */ \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE
     * Initialize sum to 0.0 */ \
    float sum = 0.0;, \
 \
    /* WEIGHT_OP
     * Calculate weight input, add to sum, update modifying substance */ \
    CALC_VAL(from_index, weight_index); \
    sum += val; \
    UPDATE_CALC;, \
 \
    /* NEURON_POST
     * Aggregate sum to input */ \
    AGGREGATE(to_index, sum); \
)

/* Vanilla versions of activator functions */
ACTIVATE_FULLY_CONNECTED(activate_fully_connected , , );
ACTIVATE_ONE_TO_ONE(activate_one_to_one , , );
ACTIVATE_CONVERGENT(activate_convergent , , );

/* Trace versions of activator functions */
ACTIVATE_FULLY_CONNECTED(activate_fully_connected_trace, EXTRACT_TRACE, UPDATE_TRACE(weight_index));
ACTIVATE_ONE_TO_ONE(activate_one_to_one_trace, EXTRACT_TRACE, UPDATE_TRACE(index));
ACTIVATE_CONVERGENT(activate_convergent_trace, EXTRACT_TRACE,
    if (convolutional) {
        UPDATE_TRACE((to_index*num_weights + weight_index));
    } else {
        UPDATE_TRACE(weight_index);
    }
);

/* Dendritic tree internal computation */
#ifdef PARALLEL
GLOBAL void calc_internal(int size, float *src, float *dst, bool clear) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        dst[index] += src[index];
        if (clear) src[index] = 0.0;
    }
}
#else
GLOBAL void calc_internal(int size, float *src, float *dst, bool clear) {
    if (clear) {
        for (int index = 0 ; index < size ; ++index) {
            dst[index] += src[index];
            src[index] = 0.0;
        }
    } else
        for (int index = 0 ; index < size ; ++index)
            dst[index] += src[index];
}
#endif
