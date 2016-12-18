#include "engine/kernel/updater_kernel.h"
#include "util/error_manager.h"

#define EXTRACT_MOD \
    float *mod = kernel_data.weights + (2*kernel_data.num_weights);

#define EXTRACT_BASELINE \
    float *baseline = kernel_data.weights + kernel_data.num_weights;

#define UPDATE_WEIGHT(weight_index, input) \
    float old_weight = kernel_data.weights[weight_index]; \
    float new_weight = old_weight + (mod[weight_index] * input * SUM_COEFFICIENT) \
                        - (WEIGHT_DECAY * (old_weight - baseline[weight_index])); \
    kernel_data.weights[weight_index] = (new_weight > kernel_data.max_weight) ? kernel_data.max_weight : new_weight;

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
        case CONVOLUTIONAL:
            return update_convergent;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

CALC_FULLY_CONNECTED(update_fully_connected, \
    /* EXTRACTIONS
     * Pointer to weight baseline
     * Pointer to modifying substance level */
    EXTRACT_BASELINE;
    EXTRACT_MOD;,

    /* NEURON_PRE
     * Retrieve input sum */
    float sum = kernel_data.inputs[to_index];,

    /* WEIGHT_OP
     * Update weight */
    UPDATE_WEIGHT(weight_index, sum);,

    /* NEURON_POST
     * no_op */
    ;
)

CALC_ONE_TO_ONE(update_one_to_one, \
    /* EXTRACTIONS
     * Pointer to weight baseline
     * Pointer to modifying substance level */
    EXTRACT_BASELINE;
    EXTRACT_MOD;,

    /* WEIGHT_OP
     * Update weight */
    UPDATE_WEIGHT(index, kernel_data.inputs[index]);
)

CALC_CONVERGENT(update_convergent, \
    /* EXTRACTIONS
     * Pointer to weight baseline
     * Pointer to modifying substance level */
    EXTRACT_BASELINE;
    EXTRACT_MOD;,

    /* NEURON_PRE
     * Retrieve input sum */
    float sum = kernel_data.inputs[to_index];,

    /* WEIGHT_OP
     * Update weight */
    UPDATE_WEIGHT(weight_index, sum);,

    /* NEURON_POST
     * no_op */
    ;
)
