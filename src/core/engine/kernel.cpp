#include "kernel.h"
#include "util/parallel.h"
#include "engine/instruction.h"
#include "util/error_manager.h"

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
#ifdef PARALLEL
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < inst.to_size) {
        float sum = 0.0;
        for (int row = 0 ; row < inst.from_size ; ++row) {
            sum += inst.extractor(inst, inst.outputs[row])
                * inst.weights[row * inst.to_size + col];
        }
        inst.inputs[col] = calc(inst.opcode, inst.inputs[col], sum);
    }
#else
    for (int row = 0 ; row < inst.to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < inst.from_size ; ++col) {
            sum += inst.extractor(inst, inst.outputs[col]) *
                inst.weights[row * inst.from_size + col];
        }
        inst.inputs[row] = calc(inst.opcode, inst.inputs[row], sum);
    }
#endif
}

GLOBAL void calc_one_to_one(Instruction inst) {
#ifdef PARALLEL
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < inst.to_size) {
#else
    for (int index = 0 ; index < inst.to_size ; ++index) {
#endif
        inst.inputs[index] = calc(inst.opcode, inst.inputs[index],
            inst.extractor(inst, inst.outputs[index]) * inst.weights[index]);
    }
}

GLOBAL void calc_convergent(Instruction inst) {
    int kernel_size = inst.overlap * inst.overlap;

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

                    sum += inst.extractor(inst, inst.outputs[s_index]) *
                        inst.weights[weight_offset + weight_col];
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

GLOBAL void update_fully_connected(Instruction inst) {
#ifdef PARALLEL
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < inst.to_size) {
        for (int row = 0 ; row < inst.from_size ; ++row) {
            /* Do stuff to weight */
        }
    }
#else
    for (int row = 0 ; row < inst.to_size ; ++row) {
        for (int col = 0 ; col < inst.from_size ; ++col) {
            /* Do stuff to weight */
        }
    }
#endif
}

GLOBAL void update_one_to_one(Instruction inst) {
#ifdef PARALLEL
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < inst.to_size) {
#else
    for (int index = 0 ; index < inst.to_size ; ++index) {
#endif
        /* Do stuff to weight */
    }
}

GLOBAL void update_convergent(Instruction inst) {
    int kernel_size = inst.overlap * inst.overlap;

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

                    /* Do stuff to weight */
                }
            }
#ifndef PARALLEL
        }
#endif
    }
}
