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
    float val = extractor(delay, outputs[from_index]) * weights[weight_index];

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

CALC_FULLY_CONNECTED(activate_fully_connected, \
    /* EXTRACTIONS
     * Pointer to activation trace */
    EXTRACT_TRACE;,

    /* NEURON_PRE
     * Initialize sum to 0.0 */
    float sum = 0.0;,

    /* WEIGHT_OP
     * Calculate weight input, add to sum, update modifying substance */
    CALC_VAL(from_index, weight_index);
    sum += val;
    UPDATE_TRACE(weight_index);,

    /* NEURON_POST
     * Aggregate sum to input */
    AGGREGATE(to_index, sum);
)

CALC_ONE_TO_ONE(activate_one_to_one, \
    /* EXTRACTIONS
     * Pointer to activation trace */
    EXTRACT_TRACE;,

    /* WEIGHT_OP
     * Calculate weight input, add to sum, update modifying substance
     * Aggregate weight input to total input */
    CALC_VAL(index, index);
    UPDATE_TRACE(index);
    AGGREGATE(index, val);
)

CALC_CONVERGENT(activate_convergent, \
    /* EXTRACTIONS
     * Pointer to activation trace */
    EXTRACT_TRACE;,

    /* NEURON_PRE
     * Initialize sum to 0.0 */
    float sum = 0.0;,

    /* WEIGHT_OP
     * Calculate weight input, add to sum, update modifying substance */
    CALC_VAL(from_index, weight_index);
    sum += val;
    if (not convolutional) UPDATE_TRACE(weight_index);,

    /* NEURON_POST
     * Aggregate sum to input */
    AGGREGATE(to_index, sum);
)
