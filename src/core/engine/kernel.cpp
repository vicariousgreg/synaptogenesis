#include "kernel.h"
#include "util/parallel.h"
#include "engine/instruction.h"
#include "util/error_manager.h"

// Device pointers for memcpyFromSymbol
DEVICE EXTRACTOR x_float = extract_float;
DEVICE EXTRACTOR x_int = extract_int;
DEVICE EXTRACTOR x_bit = extract_bit;

void get_kernel(KERNEL *dest, ConnectionType conn_type) {
    switch (conn_type) {
        case (FULLY_CONNECTED):
            *dest = calc_fully_connected;
            break;
        case (ONE_TO_ONE):
            *dest = calc_one_to_one;
            break;
        case (DIVERGENT):
        case (DIVERGENT_CONVOLUTIONAL):
            *dest = calc_divergent;
            break;
        case (CONVERGENT):
        case (CONVERGENT_CONVOLUTIONAL):
            *dest = calc_convergent;
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
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

DEVICE float extract_float(Instruction &inst, Output &out) { return out.f; }
DEVICE float extract_int(Instruction &inst, Output &out) { return out.i; }
DEVICE float extract_bit(Instruction &inst, Output &out) {
    return (out.i >> inst.delay) & 1;
}

GLOBAL void clear_data(float* data, int count) {
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count)
#else
    for (int nid = 0; nid < count; ++nid)
#endif
        data[nid] = 0.0;
}

#ifdef PARALLEL
GLOBAL void calc_fully_connected(Instruction inst) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < inst.to_size) {
        float sum = 0;
        for (int row = 0 ; row < inst.from_size ; ++row) {
            sum += inst.extractor(inst, inst.outputs[row])
                * inst.weights[row * inst.to_size + col];
        }
        inst.inputs[col] = calc(inst.opcode, inst.inputs[col], sum);
    }
}

GLOBAL void calc_one_to_one(Instruction inst) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < inst.from_size) {
        inst.inputs[index] = calc(inst.opcode, inst.inputs[index],
            inst.extractor(inst, inst.outputs[index]) * inst.weights[index]);
    }
}

GLOBAL void calc_divergent(Instruction inst) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*inst.to_columns + d_col;

    if (d_row < inst.to_rows and d_col < inst.to_columns) {
        float sum = 0.0;

        // Determine range of source neurons for divergent kernel
        int start_s_row = (d_row - inst.fray - inst.overlap + inst.stride) / inst.stride;
        int start_s_col = (d_col - inst.fray - inst.overlap + inst.stride) / inst.stride;
        int end_s_row = (d_row - inst.fray) / inst.stride;
        int end_s_col = (d_col - inst.fray) / inst.stride;

        // Kernels are organized into columns
        // One kernel per source neuron
        //   Unless convolutional (shared kernel)
        int kernel_size = inst.overlap * inst.overlap;
        int kernel_row_size = (inst.convolutional)
                              ? 1 : inst.from_rows * inst.from_columns;

        // Iterate over relevant source neurons...
        for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
            for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                // Avoid making connections with non-existent neurons!
                if (s_row < 0 or s_row >= inst.from_rows
                    or s_col < 0 or s_col >= inst.from_columns)
                    continue;
                int s_index = (s_row * inst.from_columns) + s_col;
                int k_row =
                    (d_row + ((inst.overlap - inst.stride) * s_row)
                    % inst.overlap);
                int k_col =
                    (d_col + ((inst.overlap - inst.stride) * s_col)
                    % inst.overlap);

                // Row of matrix is the kernel index * row size (see above)
                int weight_offset =
                    ((k_row * inst.overlap) + k_col)
                    * kernel_row_size;
                // Column of matrix is either the first column (convolutional)
                //   or the index of the source neuron otherwise
                int weight_col = (inst.convolutional)
                                 ? 0 : s_index;

                sum += inst.extractor(inst, inst.outputs[s_index]) *
                    inst.weights[weight_offset + weight_col];
            }
        }
        inst.inputs[d_index] = calc(inst.opcode, inst.inputs[d_index], sum);
    }
}

GLOBAL void calc_convergent(Instruction inst) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*inst.to_columns + d_col;

    if (d_row < inst.to_rows and d_col < inst.to_columns) {
        float sum = 0.0;
        int s_row = d_row * inst.stride - inst.fray;
        int s_col = d_col * inst.stride - inst.fray;

        // Kernels are organized into columns
        // One kernel per destination neuron
        //   Unless convolutional (shared kernel)
        int kernel_size = inst.overlap * inst.overlap;
        int kernel_row_size = (inst.convolutional)
                              ? 1 : inst.to_rows * inst.to_columns;

        // Column of matrix is either the first column (convolutional)
        //   or the index of the destination neuron otherwise
        int weight_col = (inst.convolutional) ? 0 : d_index;

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

                // Row of matrix is the kernel index * row size (see above)
                int weight_offset =
                    ((k_row*inst.overlap) + k_col)
                    * kernel_row_size;

                sum += inst.extractor(inst, inst.outputs[s_index]) *
                    inst.weights[weight_offset + weight_col];
            }
        }
        inst.inputs[d_index] = calc(inst.opcode, inst.inputs[d_index], sum);
    }
}
#else

/* IMPORTANT:
 * Serial implementation is faster if matrix is interpreted in a transposed
 *    fashion compared to parallel.  In this loop, row is the destination,
 *    column is the source.  In this way, inputs to one neuron are
 *    contiguous in memory.
 */

void calc_fully_connected(Instruction inst) {
    for (int row = 0 ; row < inst.to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < inst.from_size ; ++col) {
            sum += inst.extractor(inst, inst.outputs[col]) *
                inst.weights[row*inst.from_size + col];
        }
        inst.inputs[row] = calc(inst.opcode, inst.inputs[row], sum);
    }
}

void calc_one_to_one(Instruction inst) {
    for (int index = 0 ; index < inst.from_size ; ++index) {
        inst.inputs[index] = calc(inst.opcode, inst.inputs[index],
            inst.extractor(inst, inst.outputs[index]) * inst.weights[index]);
    }
}

void calc_divergent(Instruction inst) {
    int kernel_size = inst.overlap * inst.overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < inst.to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < inst.to_columns ; ++d_col) {
            int d_index = d_row*inst.to_columns + d_col;
            float sum = 0.0;

            // Determine range of source neurons for divergent kernel
            int start_s_row = (d_row - inst.fray - inst.overlap + inst.stride) / inst.stride;
            int start_s_col = (d_col - inst.fray - inst.overlap + inst.stride) / inst.stride;
            int end_s_row = (d_row - inst.fray) / inst.stride;
            int end_s_col = (d_col - inst.fray) / inst.stride;

            // Iterate over relevant source neurons...
            for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
                for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                    // Avoid making connections with non-existent neurons!
                    if (s_row < 0 or s_row >= inst.from_rows
                        or s_col < 0 or s_col >= inst.from_columns)
                        continue;

                    int s_index = (s_row * inst.from_columns) + s_col;
                    int k_row =
                        (d_row + ((inst.overlap - inst.stride) * s_row)
                        % inst.overlap);
                    int k_col =
                        (d_col + ((inst.overlap - inst.stride) * s_col)
                        % inst.overlap);

                    // Row of matrix is either the first column (convolutional)
                    //   or the index of the source neuron otherwise
                    int weight_offset = (inst.convolutional)
                                        ? 0 : s_index * kernel_size;
                    // Column of matrix is the kernel index
                    int k_index = (k_row * inst.overlap) + k_col;

                    sum += inst.extractor(inst, inst.outputs[s_index]) *
                        inst.weights[weight_offset + k_index];
                }
            }
            inst.inputs[d_index] = calc(inst.opcode, inst.inputs[d_index], sum);
        }
    }
}

void calc_convergent(Instruction inst) {
    int kernel_size = inst.overlap * inst.overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < inst.to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < inst.to_columns ; ++d_col) {
            int d_index = d_row * inst.to_columns + d_col;
            float sum = 0.0;

            // Determine starting row and column for source neurons
            int s_row = d_row * inst.stride - inst.fray;
            int s_col = d_col * inst.stride - inst.fray;

            // Row of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_offset = (inst.convolutional)
                                ? 0 : d_index * kernel_size;

            // Run the kernel
            for (int k_row = 0 ; k_row < inst.overlap ; ++k_row) {
                for (int k_col = 0 ; k_col < inst.overlap ; ++k_col) {
                    int k_s_row = s_row + k_row;
                    int k_s_col = s_col + k_col;
                    // The connection is frayed if the layers are the same size
                    // Avoid making connections with non-existent neurons!
                    if (inst.fray > 0 and (k_s_row < 0 or k_s_row >= inst.to_rows
                        or k_s_col < 0 or k_s_col >= inst.to_columns))
                        continue;

                    int s_index = k_s_row * inst.from_columns + k_s_col;

                    // Column of matrix is the kernel index
                    int weight_col = (k_row * inst.overlap) + k_col;
                    sum += inst.extractor(inst, inst.outputs[s_index]) *
                        inst.weights[weight_offset + weight_col];
                }
            }
            inst.inputs[d_index] = calc(inst.opcode, inst.inputs[d_index], sum);
        }
    }
}
#endif
