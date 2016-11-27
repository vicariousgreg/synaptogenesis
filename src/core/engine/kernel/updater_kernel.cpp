#include "engine/kernel/updater_kernel.h"
#include "util/error_manager.h"

#define UPDATE_WEIGHT(index, input) \
    float old_weight = kernel_data.weights[index]; \
    float new_weight = old_weight + (mod[index] * input * SUM_COEFFICIENT) \
                        - (WEIGHT_DECAY * (old_weight - baseline[index])); \
    kernel_data.weights[index] = (new_weight > kernel_data.max_weight) ? kernel_data.max_weight : new_weight;

/******************************************************************************/
/********************** CONNECTION UPDATER KERNELS ****************************/
/******************************************************************************/

void get_updater(UPDATER *dest, ConnectionType conn_type) {
    switch (conn_type) {
        case (FULLY_CONNECTED):
            *dest = update_fully_connected;
            break;
        case (ONE_TO_ONE):
            *dest = update_one_to_one;
            break;
        case (CONVERGENT):
        case (CONVOLUTIONAL):
            *dest = update_convergent;
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

CALC_FULLY_CONNECTED(update_fully_connected, \
    // Pointer to modifying substance level and weight baseline
    float *baseline = kernel_data.weights + kernel_data.num_weights;
    float *mod = kernel_data.weights + (2*kernel_data.num_weights);,

    // Retrieve input sum
    float sum = kernel_data.inputs[to_index];,

    // Update weight
    UPDATE_WEIGHT(weight_index, sum);,

    ;
)

CALC_ONE_TO_ONE(update_one_to_one, \
    // Pointer to modifying substance level and weight baseline
    float *baseline = kernel_data.weights + kernel_data.num_weights;
    float *mod = kernel_data.weights + (2*kernel_data.num_weights);,

    // Update weight
    UPDATE_WEIGHT(index, kernel_data.inputs[index]);
)

CALC_CONVERGENT(update_convergent, \
    // Pointer to modifying substance level and weight baseline
    float *baseline = kernel_data.weights + kernel_data.num_weights;
    float *mod = kernel_data.weights + (2*kernel_data.num_weights);,

    // Retrieve input sum
    float sum = kernel_data.inputs[to_index];,

    // Update weight
    UPDATE_WEIGHT(weight_index, sum);,

    ;
)
