#include <cmath>
#include "kernel.h"
#include "util/parallel.h"
#include "engine/instruction.h"
#include "util/error_manager.h"


#define MOD_RATE 0.3
#define MOD_DECAY 0.01
#define MOD_MAX 10.0
#define SUM_COEFFICIENT 0.5
#define WEIGHT_DECAY 0.025

/*
#define MOD_RATE 0.3
#define MOD_DECAY 0.01
#define MOD_MAX 10.0
#define SUM_COEFFICIENT 0.5
#define WEIGHT_DECAY 0.025
*/


/******************************************************************************/
/************************** OUTPUT EXTRACTORS *********************************/
/******************************************************************************/

// Device pointers for memcpyFromSymbol
DEVICE EXTRACTOR x_float = extract_float;
DEVICE EXTRACTOR x_int = extract_int;
DEVICE EXTRACTOR x_bit = extract_bit;

DEVICE float extract_float(Instruction &inst, Output &out) { return out.f; }
DEVICE float extract_int(Instruction &inst, Output &out) { return out.i; }
DEVICE float extract_bit(Instruction &inst, Output &out) {
    return (out.i >> (inst.delay % 32)) & 1;
}

void get_extractor(EXTRACTOR *dest, OutputType output_type) {
#ifdef PARALLEL
    switch (output_type) {
        case FLOAT:
            cudaMemcpyFromSymbol(dest, x_float, sizeof(void *));
            break;
        case INT:
            cudaMemcpyFromSymbol(dest, x_int, sizeof(void *));
            break;
        case BIT:
            cudaMemcpyFromSymbol(dest, x_bit, sizeof(void *));
            break;
    }
#else
    switch (output_type) {
        case FLOAT:
            *dest = extract_float;
            break;
        case INT:
            *dest = extract_int;
            break;
        case BIT:
            *dest = extract_bit;
            break;
    }
#endif
}

/******************************************************************************/
/**************************** DATA CLEARING ***********************************/
/******************************************************************************/

GLOBAL void clear_data(float* data, int count) {
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count)
#else
    for (int nid = 0; nid < count; ++nid)
#endif
        data[nid] = 0.0;
}

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

GLOBAL void calc_fully_connected(Instruction inst) {
    // Pointer to modifying substance level
    float *mod = inst.weights + (2*inst.num_weights);

#ifdef PARALLEL
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < inst.to_size) {
        float sum = 0.0;
        for (int row = 0 ; row < inst.from_size ; ++row) {
            int index = row * inst.to_size + col;
            float val = inst.extractor(inst, inst.outputs[row])
                        * inst.weights[index];
            sum += val;

            // If plastic, update modifying substance
            if (inst.plastic) {
                float old_mod = mod[index];
                float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
                mod[index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
            }
        }
        inst.inputs[col] = calc(inst.opcode, inst.inputs[col], sum);
    }
#else
    for (int row = 0 ; row < inst.to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < inst.from_size ; ++col) {
            int index = row * inst.from_size + col;
            float val = inst.extractor(inst, inst.outputs[col])
                        * inst.weights[index];
            sum += val;

            // If plastic, update modifying substance
            if (inst.plastic) {
                float old_mod = mod[index];
                float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
                mod[index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
            }

        }
        inst.inputs[row] = calc(inst.opcode, inst.inputs[row], sum);
    }
#endif
}

GLOBAL void calc_one_to_one(Instruction inst) {
    // Pointer to modifying substance level
    float *mod = inst.weights + (2*inst.num_weights);

#ifdef PARALLEL
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < inst.to_size) {
#else
    for (int index = 0 ; index < inst.to_size ; ++index) {
#endif
        float val = calc(inst.opcode, inst.inputs[index],
            inst.extractor(inst, inst.outputs[index]) * inst.weights[index]);
        inst.inputs[index] = val;

        // Update modifying substance
        // If plastic, update modifying substance
        if (inst.plastic) {
            float old_mod = mod[index];
            float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
            mod[index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
        }
    }
}

GLOBAL void calc_convergent(Instruction inst) {
    // Pointer to modifying substance level
    float *mod = inst.weights + (2*inst.num_weights);

    int kernel_size = inst.overlap * inst.overlap;

#ifdef PARALLEL
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    if (d_row < inst.to_rows and d_col < inst.to_columns) {
#else
    for (int d_row = 0 ; d_row < inst.to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < inst.to_columns ; ++d_col) {
#endif
            int d_index = d_row * inst.to_columns + d_col;
            float sum = 0.0;

            // Determine starting row and column for source neurons
            int s_row = d_row * inst.stride - inst.fray;
            int s_col = d_col * inst.stride - inst.fray;

#ifdef PARALLEL
            // Column of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_col = (inst.convolutional) ? 0 : d_index;
            // Kernels are organized into columns
            // One kernel per destination neuron
            //   Unless convolutional (shared kernel)
            int kernel_row_size = (inst.convolutional)
                                  ? 1 : inst.to_size;
#else
            // Row of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_offset = (inst.convolutional)
                                ? 0 : d_index * kernel_size;
#endif

            // Run the kernel
            for (int k_row = 0 ; k_row < inst.overlap ; ++k_row) {
                for (int k_col = 0 ; k_col < inst.overlap ; ++k_col) {
                    int k_s_row = s_row + k_row;
                    int k_s_col = s_col + k_col;

                    // The connection is frayed if the layers are the same size
                    // Avoid making connections with non-existent neurons!
                    if (inst.fray != 0 and (k_s_row < 0 or k_s_row >= inst.to_rows
                        or k_s_col < 0 or k_s_col >= inst.to_columns))
                        continue;

                    int s_index = k_s_row * inst.from_columns + k_s_col;

#ifdef PARALLEL
                    // Row of matrix is the kernel index * row size (see above)
                    int weight_offset =
                        ((k_row*inst.overlap) + k_col)
                        * kernel_row_size;
#else
                    // Column of matrix is the kernel index
                    int weight_col = (k_row * inst.overlap) + k_col;
#endif
                    int weight_index = weight_offset + weight_col;

                    float val = inst.extractor(inst, inst.outputs[s_index])
                                    * inst.weights[weight_index];
                    sum += val;

                    // Update modifying substance
                    // If plastic, update modifying substance
                    if (inst.plastic and not inst.convolutional) {
                        float old_mod = mod[weight_index];
                        float new_mod = old_mod + (MOD_RATE * val) - (MOD_DECAY * old_mod);
                        mod[weight_index] = (new_mod < 0.0) ? 0.0 : (new_mod > MOD_MAX) ? MOD_MAX : new_mod;
                    }
                }
            }
            inst.inputs[d_index] = calc(inst.opcode, inst.inputs[d_index], sum);
#ifndef PARALLEL
        }
#endif
    }
}

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

GLOBAL void update_fully_connected(Instruction inst) {
    // Pointer to modifying substance level and weight baseline
    float *baseline = inst.weights + inst.num_weights;
    float *mod = inst.weights + (2*inst.num_weights);

#ifdef PARALLEL
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < inst.to_size) {
        float sum = inst.inputs[col];
        for (int row = 0 ; row < inst.from_size ; ++row) {
            int index = row * inst.to_size + col;

            // Update weight
            float old_weight = inst.weights[index];
            float new_weight = old_weight + (mod[index] * sum * SUM_COEFFICIENT)
                                - (WEIGHT_DECAY * (old_weight - baseline[index]));
            inst.weights[index] = (new_weight > inst.max_weight) ? inst.max_weight : new_weight;
        }
    }
#else
    for (int row = 0 ; row < inst.to_size ; ++row) {
        float sum = inst.inputs[row];
        for (int col = 0 ; col < inst.from_size ; ++col) {
            int index = row * inst.from_size + col;

            // Update weight
            float old_weight = inst.weights[index];
            float new_weight = old_weight + (mod[index] * sum * SUM_COEFFICIENT)
                                - (WEIGHT_DECAY * (old_weight - baseline[index]));
            inst.weights[index] = (new_weight > inst.max_weight) ? inst.max_weight : new_weight;
            //if (old_weight != new_weight) printf("(%10f ->  %10f    %c )\n", old_weight, new_weight, (new_weight > old_weight) ? '+' : '-');
        }
    }
#endif
}

GLOBAL void update_one_to_one(Instruction inst) {
    // Pointer to modifying substance level and weight baseline
    float *baseline = inst.weights + inst.num_weights;
    float *mod = inst.weights + (2*inst.num_weights);

#ifdef PARALLEL
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < inst.to_size) {
#else
    for (int index = 0 ; index < inst.to_size ; ++index) {
#endif
        // Update weight
        float old_weight = inst.weights[index];
        float new_weight = old_weight + (mod[index] * inst.inputs[index] * SUM_COEFFICIENT)
                            - (WEIGHT_DECAY * (old_weight - baseline[index]));
        inst.weights[index] = (new_weight > inst.max_weight) ? inst.max_weight : new_weight;
    }
}

GLOBAL void update_convergent(Instruction inst) {
    int kernel_size = inst.overlap * inst.overlap;

    // Pointer to modifying substance level and weight baseline
    float *baseline = inst.weights + inst.num_weights;
    float *mod = inst.weights + (2*inst.num_weights);

#ifdef PARALLEL
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*inst.to_columns + d_col;
    if (d_row < inst.to_rows and d_col < inst.to_columns) {
#else
    for (int d_row = 0 ; d_row < inst.to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < inst.to_columns ; ++d_col) {
            int d_index = d_row * inst.to_columns + d_col;
#endif
            float sum = inst.inputs[d_index];

            // Determine starting row and column for source neurons
            int s_row = d_row * inst.stride - inst.fray;
            int s_col = d_col * inst.stride - inst.fray;

#ifdef PARALLEL
            // Column of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_col = (inst.convolutional) ? 0 : d_index;
            // Kernels are organized into columns
            // One kernel per destination neuron
            //   Unless convolutional (shared kernel)
            int kernel_row_size = (inst.convolutional)
                                  ? 1 : inst.to_rows * inst.to_columns;
#else
            // Row of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_offset = (inst.convolutional)
                                ? 0 : d_index * kernel_size;
#endif

            // Run the kernel
            for (int k_row = 0 ; k_row < inst.overlap ; ++k_row) {
                for (int k_col = 0 ; k_col < inst.overlap ; ++k_col) {
                    int k_s_row = s_row + k_row;
                    int k_s_col = s_col + k_col;

                    // The connection is frayed if the layers are the same size
                    // Avoid making connections with non-existent neurons!
                    if (inst.fray != 0 and (k_s_row < 0 or k_s_row >= inst.to_rows
                        or k_s_col < 0 or k_s_col >= inst.to_columns))
                        continue;

                    int s_index = k_s_row * inst.from_columns + k_s_col;

#ifdef PARALLEL
                    // Row of matrix is the kernel index * row size (see above)
                    int weight_offset =
                        ((k_row*inst.overlap) + k_col)
                        * kernel_row_size;
#else
                    // Column of matrix is the kernel index
                    int weight_col = (k_row * inst.overlap) + k_col;
#endif
                    int weight_index = weight_offset + weight_col;

                    // Update weight
                    float old_weight = inst.weights[weight_index];
                    float new_weight = old_weight + (mod[weight_index] * sum * SUM_COEFFICIENT)
                                        - (WEIGHT_DECAY * (old_weight - baseline[weight_index]));
                    inst.weights[weight_index] = (new_weight > inst.max_weight) ? inst.max_weight : new_weight;
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
