#include <cmath>
#include "engine/kernel/activator_kernel.h"
#include "engine/instruction.h"
#include "util/parallel.h"
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

GLOBAL void calc_fully_connected(ConnectionData conn_data) {
    // Pointer to modifying substance level
    float *mod = conn_data.weights + (2*conn_data.num_weights);

#ifdef PARALLEL
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < conn_data.to_size) {
        float sum = 0.0;
        for (int row = 0 ; row < conn_data.from_size ; ++row) {
            int index = row * conn_data.to_size + col;
            float val = conn_data.extractor(conn_data, conn_data.outputs[row])
                        * conn_data.weights[index];
            sum += val;

            // If plastic, update modifying substance
            if (conn_data.plastic) {
                float old_mod = mod[index];
                float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
                mod[index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
            }
        }
        conn_data.inputs[col] = calc(conn_data.opcode, conn_data.inputs[col], sum);
    }
#else
    for (int row = 0 ; row < conn_data.to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < conn_data.from_size ; ++col) {
            int index = row * conn_data.from_size + col;
            float val = conn_data.extractor(conn_data, conn_data.outputs[col])
                        * conn_data.weights[index];
            sum += val;

            // If plastic, update modifying substance
            if (conn_data.plastic) {
                float old_mod = mod[index];
                float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
                mod[index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
            }

        }
        conn_data.inputs[row] = calc(conn_data.opcode, conn_data.inputs[row], sum);
    }
#endif
}

GLOBAL void calc_one_to_one(ConnectionData conn_data) {
    // Pointer to modifying substance level
    float *mod = conn_data.weights + (2*conn_data.num_weights);

#ifdef PARALLEL
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < conn_data.to_size) {
#else
    for (int index = 0 ; index < conn_data.to_size ; ++index) {
#endif
        float val = calc(conn_data.opcode, conn_data.inputs[index],
            conn_data.extractor(conn_data, conn_data.outputs[index]) * conn_data.weights[index]);
        conn_data.inputs[index] = val;

        // Update modifying substance
        // If plastic, update modifying substance
        if (conn_data.plastic) {
            float old_mod = mod[index];
            float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
            mod[index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
        }
    }
}

GLOBAL void calc_convergent(ConnectionData conn_data) {
    // Pointer to modifying substance level
    float *mod = conn_data.weights + (2*conn_data.num_weights);

    int kernel_size = conn_data.overlap * conn_data.overlap;

#ifdef PARALLEL
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    if (d_row < conn_data.to_rows and d_col < conn_data.to_columns) {
#else
    for (int d_row = 0 ; d_row < conn_data.to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < conn_data.to_columns ; ++d_col) {
#endif
            int d_index = d_row * conn_data.to_columns + d_col;
            float sum = 0.0;

            // Determine starting row and column for source neurons
            int s_row = d_row * conn_data.stride - conn_data.fray;
            int s_col = d_col * conn_data.stride - conn_data.fray;

#ifdef PARALLEL
            // Column of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_col = (conn_data.convolutional) ? 0 : d_index;
            // Kernels are organized into columns
            // One kernel per destination neuron
            //   Unless convolutional (shared kernel)
            int kernel_row_size = (conn_data.convolutional)
                                  ? 1 : conn_data.to_size;
#else
            // Row of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_offset = (conn_data.convolutional)
                                ? 0 : d_index * kernel_size;
#endif

            // Run the kernel
            for (int k_row = 0 ; k_row < conn_data.overlap ; ++k_row) {
                for (int k_col = 0 ; k_col < conn_data.overlap ; ++k_col) {
                    int k_s_row = s_row + k_row;
                    int k_s_col = s_col + k_col;

                    // The connection is frayed if the layers are the same size
                    // Avoid making connections with non-existent neurons!
                    if (conn_data.fray != 0 and (k_s_row < 0 or k_s_row >= conn_data.to_rows
                        or k_s_col < 0 or k_s_col >= conn_data.to_columns))
                        continue;

                    int s_index = k_s_row * conn_data.from_columns + k_s_col;

#ifdef PARALLEL
                    // Row of matrix is the kernel index * row size (see above)
                    int weight_offset =
                        ((k_row*conn_data.overlap) + k_col)
                        * kernel_row_size;
#else
                    // Column of matrix is the kernel index
                    int weight_col = (k_row * conn_data.overlap) + k_col;
#endif
                    int weight_index = weight_offset + weight_col;

                    float val = conn_data.extractor(conn_data, conn_data.outputs[s_index])
                                    * conn_data.weights[weight_index];
                    sum += val;

                    // Update modifying substance
                    // If plastic, update modifying substance
                    if (conn_data.plastic and not conn_data.convolutional) {
                        float old_mod = mod[weight_index];
                        float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
                        mod[weight_index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
                    }
                }
            }
            conn_data.inputs[d_index] = calc(conn_data.opcode, conn_data.inputs[d_index], sum);
#ifndef PARALLEL
        }
#endif
    }
}
