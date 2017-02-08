#include "engine/kernel/updater_kernel.h"
#include "state/attributes.h"
#include "util/error_manager.h"

#define SUM_COEFFICIENT 0.01
#define WEIGHT_DECAY 0.0001

#define EXTRACT_TRACE \
    float *trace = weights + (2*num_weights);

#define EXTRACT_BASELINE \
    float *baseline = weights + num_weights;

#define UPDATE_WEIGHT(weight_index, input) \
    float old_weight = weights[weight_index]; \
    float new_weight = old_weight + (trace[weight_index] * input * SUM_COEFFICIENT) \
                        - (WEIGHT_DECAY * (old_weight - baseline[weight_index])); \
    weights[weight_index] = (new_weight > max_weight) ? max_weight : new_weight;

#define UPDATE_WEIGHT_CONVOLUTIONAL(weight_index, input) \
    float old_weight = weights[weight_index]; \
    float t = 0.0; \
    for (int i = 0; i < to_size; ++i) { \
        t += trace[i*num_weights + weight_index]; \
    } \
    t /= to_size; \
    float new_weight = old_weight + (t * input * SUM_COEFFICIENT) \
                        - (WEIGHT_DECAY * (old_weight - baseline[weight_index])); \
    weights[weight_index] = (new_weight > max_weight) ? max_weight : new_weight;

/******************************************************************************/
/********************** CONNECTION UPDATER KERNELS ****************************/
/******************************************************************************/

KERNEL get_updater_kernel(ConnectionType conn_type) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return update_fully_connected;
        case ONE_TO_ONE:
            return update_one_to_one;
        case CONVERGENT:
            return update_convergent;
        case CONVOLUTIONAL:
            return update_convolutional;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

CALC_FULLY_CONNECTED(update_fully_connected, \
    /* EXTRACTIONS
     * Pointer to activation trace
     * Pointer to weight baseline */
    EXTRACT_TRACE;
    EXTRACT_BASELINE;,

    /* NEURON_PRE
     * Retrieve input sum */
    float sum = inputs[to_index];,

    /* WEIGHT_OP
     * Update weight */
    UPDATE_WEIGHT(weight_index, sum);,

    /* NEURON_POST
     * no_op */
    ;
)

CALC_ONE_TO_ONE(update_one_to_one, \
    /* EXTRACTIONS
     * Pointer to activation trace
     * Pointer to weight baseline */
    EXTRACT_TRACE;
    EXTRACT_BASELINE;,

    /* WEIGHT_OP
     * Update weight */
    UPDATE_WEIGHT(index, inputs[index]);
)

CALC_CONVERGENT(update_convergent, \
    /* EXTRACTIONS
     * Pointer to activation trace
     * Pointer to weight baseline */
    EXTRACT_TRACE;
    EXTRACT_BASELINE;,

    /* NEURON_PRE
     * Retrieve input sum */
    float sum = inputs[to_index];,

    /* WEIGHT_OP
     * Update weight */
    UPDATE_WEIGHT(weight_index, sum);,

    /* NEURON_POST
     * no_op */
    ;
)

CALC_ONE_TO_ONE(update_convolutional, \
    /* EXTRACTIONS
     * Pointer to activation trace
     * Pointer to weight baseline */
    EXTRACT_TRACE;
    EXTRACT_BASELINE;,

    /* WEIGHT_OP
     * Update weight */
    UPDATE_WEIGHT_CONVOLUTIONAL(index, inputs[index]);
)
