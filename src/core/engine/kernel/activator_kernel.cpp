#include "engine/kernel/activator_kernel.h"
#include "util/error_manager.h"

/******************************************************************************/
/********************** CONNECTION ACTIVATOR KERNELS **************************/
/******************************************************************************/

/* Parallel and Serial kernels are combined using preprocessor directives.
 * The Parallel versions determine an index based on the CUDA thread/block.
 * The Serial versions perform iterations over all of the neurons.
 *
 * IMPORTANT:
 * Serial implementation is faster if matrix is interpreted in a transposed
 *    fashion compared to parallel.  For serial loops, row is the destination,
 *    column is the source.  This way, inputs to one neuron are contiguous in
 *    memory.
 */

void get_activator(ACTIVATOR *dest, ConnectionType conn_type) {
    switch (conn_type) {
        case (FULLY_CONNECTED):
            *dest = calc_fully_connected;
            break;
        case (ONE_TO_ONE):
            *dest = calc_one_to_one;
            break;
        case (CONVERGENT):
        case (CONVOLUTIONAL):
            *dest = calc_convergent;
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

CALC_FULLY_CONNECTED(calc_fully_connected, \
    /* Pointer to modifying substance level */
    float *mod = kernel_data.weights + (2*kernel_data.num_weights);,

    float sum = 0.0;,

    float val = kernel_data.extractor(kernel_data, kernel_data.outputs[from_index])
                * kernel_data.weights[weight_index];
    sum += val;
    /* If plastic, update modifying substance */
    if (kernel_data.plastic) {
        float old_mod = mod[weight_index];
        float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
        mod[weight_index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
    },

    kernel_data.inputs[to_index] = calc(kernel_data.opcode, kernel_data.inputs[to_index], sum);
)

CALC_ONE_TO_ONE(calc_one_to_one, \
    /* Pointer to modifying substance level */
    float *mod = kernel_data.weights + (2*kernel_data.num_weights);,

    float val = calc(kernel_data.opcode, kernel_data.inputs[index],
        kernel_data.extractor(kernel_data, kernel_data.outputs[index]) * kernel_data.weights[index]);
    // Update modifying substance
    // If plastic, update modifying substance
    if (kernel_data.plastic) {
        float old_mod = mod[index];
        float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
        mod[index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
    }
    kernel_data.inputs[index] = val;
)

CALC_CONVERGENT(calc_convergent, \
    /* Pointer to modifying substance level */
    float *mod = kernel_data.weights + (2*kernel_data.num_weights);,

    float sum = 0.0;,

    float val = kernel_data.extractor(kernel_data, kernel_data.outputs[from_index])
                    * kernel_data.weights[weight_index];
    sum += val;
    /* Update modifying substance
       If plastic, update modifying substance */
    if (kernel_data.plastic and not kernel_data.convolutional) {
        float old_mod = mod[weight_index];
        float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
        mod[weight_index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
    },

    kernel_data.inputs[to_index] = calc(kernel_data.opcode, kernel_data.inputs[to_index], sum);
)
