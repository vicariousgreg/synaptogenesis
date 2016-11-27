#include <cmath>
#include "engine/kernel/updater_kernel.h"
#include "engine/instruction.h"
#include "util/parallel.h"
#include "util/error_manager.h"

#define UPDATE_WEIGHT(index, input) \
    float old_weight = conn_data.weights[index]; \
    float new_weight = old_weight + (mod[index] * input * SUM_COEFFICIENT) \
                        - (WEIGHT_DECAY * (old_weight - baseline[index])); \
    conn_data.weights[index] = (new_weight > conn_data.max_weight) ? conn_data.max_weight : new_weight;

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

#include <iostream>

GLOBAL void update_fully_connected(ConnectionData conn_data) {
    // Pointer to modifying substance level and weight baseline
    float *baseline = conn_data.weights + conn_data.num_weights;
    float *mod = conn_data.weights + (2*conn_data.num_weights);

#ifdef PARALLEL
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < conn_data.to_size) {
        float sum = conn_data.inputs[col];
        for (int row = 0 ; row < conn_data.from_size ; ++row) {
            int index = row * conn_data.to_size + col;

            // Update weight
            UPDATE_WEIGHT(index, sum);
        }
    }
#else
    for (int row = 0 ; row < conn_data.to_size ; ++row) {
        float sum = conn_data.inputs[row];
        for (int col = 0 ; col < conn_data.from_size ; ++col) {
            int index = row * conn_data.from_size + col;

            // Update weight
            UPDATE_WEIGHT(index, sum);
            //if (old_weight != new_weight) printf("(%10f ->  %10f    %c )\n", old_weight, new_weight, (new_weight > old_weight) ? '+' : '-');
        }
    }
#endif
}

GLOBAL void update_one_to_one(ConnectionData conn_data) {
    // Pointer to modifying substance level and weight baseline
    float *baseline = conn_data.weights + conn_data.num_weights;
    float *mod = conn_data.weights + (2*conn_data.num_weights);

#ifdef PARALLEL
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < conn_data.to_size) {
#else
    for (int index = 0 ; index < conn_data.to_size ; ++index) {
#endif
        // Update weight
        UPDATE_WEIGHT(index, conn_data.inputs[index]);
    }
}

GLOBAL void update_convergent(ConnectionData conn_data) {
    int kernel_size = conn_data.overlap * conn_data.overlap;

    // Pointer to modifying substance level and weight baseline
    float *baseline = conn_data.weights + conn_data.num_weights;
    float *mod = conn_data.weights + (2*conn_data.num_weights);

#ifdef PARALLEL
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*conn_data.to_columns + d_col;
    if (d_row < conn_data.to_rows and d_col < conn_data.to_columns) {
#else
    for (int d_row = 0 ; d_row < conn_data.to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < conn_data.to_columns ; ++d_col) {
            int d_index = d_row * conn_data.to_columns + d_col;
#endif
            float sum = conn_data.inputs[d_index];

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
                                  ? 1 : conn_data.to_rows * conn_data.to_columns;
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

                    // Update weight
                    UPDATE_WEIGHT(weight_index, sum);
                    /*
                    if (weight_index == 19) // and old_weight > new_weight)
                        printf("(%d    %10f ->  %10f    %c )\n", weight_index, old_weight, new_weight,
                                                (new_weight > old_weight) ? '+' : '-');
                    */
                }
            }
#ifndef PARALLEL
        }
#endif
    }
}
