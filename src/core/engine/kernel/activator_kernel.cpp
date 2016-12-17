#include "engine/kernel/activator_kernel.h"
#include "util/error_manager.h"

#define EXTRACT_MOD \
    float *mod = kernel_data.weights + (2*kernel_data.num_weights);

#define CALC_VAL(from_index, weight_index) \
    float val = kernel_data.extractor(kernel_data, kernel_data.outputs[from_index]) \
                * kernel_data.weights[weight_index];

#define UPDATE_MOD(weight_index) \
    if (kernel_data.plastic) { \
        float old_mod = mod[weight_index]; \
        float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod); \
        mod[weight_index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod; \
    }

#define AGGREGATE(to_index, sum) \
    kernel_data.inputs[to_index] = calc(kernel_data.opcode, kernel_data.inputs[to_index], sum);

/******************************************************************************/
/********************** CONNECTION ACTIVATOR KERNELS **************************/
/******************************************************************************/

void get_activator_kernel(KERNEL *dest, ConnectionType conn_type) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            *dest = activate_fully_connected;
            break;
        case ONE_TO_ONE:
            *dest = activate_one_to_one;
            break;
        case CONVERGENT:
        case CONVOLUTIONAL:
            *dest = activate_convergent;
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

CALC_FULLY_CONNECTED(activate_fully_connected, \
    /* EXTRACTIONS
     * Pointer to modifying substance level */
    EXTRACT_MOD;,

    /* NEURON_PRE
     * Initialize sum to 0.0 */
    float sum = 0.0;,

    /* WEIGHT_OP
     * Calculate weight input, add to sum, update modifying substance */
    CALC_VAL(from_index, weight_index);
    sum += val;
    UPDATE_MOD(weight_index);,

    /* NEURON_POST
     * Aggregate sum to input */
    AGGREGATE(to_index, sum);
)

CALC_ONE_TO_ONE(activate_one_to_one, \
    /* EXTRACTIONS
     * Pointer to modifying substance level */
    EXTRACT_MOD;,

    /* WEIGHT_OP
     * Calculate weight input, add to sum, update modifying substance
     * Aggregate weight input to total input */
    CALC_VAL(index, index);
    UPDATE_MOD(index);
    AGGREGATE(index, val);
)

CALC_CONVERGENT(activate_convergent, \
    /* EXTRACTIONS
     * Pointer to modifying substance level */
    EXTRACT_MOD;,

    /* NEURON_PRE
     * Initialize sum to 0.0 */
    float sum = 0.0;,

    /* WEIGHT_OP
     * Calculate weight input, add to sum, update modifying substance */
    CALC_VAL(from_index, weight_index);
    sum += val;
    if (not kernel_data.convolutional) UPDATE_MOD(weight_index);,

    /* NEURON_POST
     * Aggregate sum to input */
    AGGREGATE(to_index, sum);
)
