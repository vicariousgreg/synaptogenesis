#ifndef kernel_h
#define kernel_h

#include "model/model.h"
#include "parallel.h"

#ifdef PARALLEL
template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < conn.to_size) {
        float sum = 0;
        for (int row = 0 ; row < conn.from_size ; ++row) {
            sum += func(outputs[row], args...) * weights[row * conn.to_size + col];
        }
        inputs[col] = calc(conn.opcode, inputs[col], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_divergent(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*conn.to_columns + d_col;

    if (d_row < conn.to_rows and d_col < conn.to_columns) {
        float sum = 0.0;

        // Determine range of source neurons for divergent kernel
        int start_s_row = d_row / conn.overlap;
        int start_s_col = d_col / conn.overlap ;
        int end_s_row = (d_row + conn.stride) / conn.overlap;
        int end_s_col = (d_col + conn.stride) / conn.overlap ;

        // Kernels are organized into columns
        // One kernel per source neuron
        //   Unless convolutional (shared kernel)
        int kernel_size = conn.overlap * conn.overlap;
        int kernel_row_size = (conn.convolutional) ? 1 : conn.from_rows * conn.from_columns;

        // Iterate over relevant source neurons...
        for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
            for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                int s_index = (s_row * conn.from_columns) + s_col;
                int k_row = (d_row + ((conn.overlap - conn.stride) * s_row) % conn.overlap);
                int k_col = (d_col + ((conn.overlap - conn.stride) * s_col) % conn.overlap);

                // Row of matrix is the kernel index * row size (see above)
                int weight_offset = ((k_row * conn.overlap) + k_col) * kernel_row_size;
                // Column of matrix is either the first column (convolutional)
                //   or the index of the source neuron otherwise
                int weight_col = (conn.convolutional) ? 0 : s_index;

                sum += func(outputs[s_index], args...) *
                    weights[weight_offset + weight_col];
            }
        }
        inputs[d_index] = calc(conn.opcode, inputs[d_index], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_convergent(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*conn.to_columns + d_col;

    if (d_row < conn.to_rows and d_col < conn.to_columns) {
        float sum = 0.0;
        int s_row = d_row * conn.stride;
        int s_col = d_col * conn.stride;

        // Kernels are organized into columns
        // One kernel per destination neuron
        //   Unless convolutional (shared kernel)
        int kernel_size = conn.overlap * conn.overlap;
        int kernel_row_size = (conn.convolutional) ? 1 : conn.to_rows * conn.to_columns;

        // Column of matrix is either the first column (convolutional)
        //   or the index of the destination neuron otherwise
        int weight_col = (conn.convolutional) ? 0 : d_index;

        // Run the kernel
        for (int k_row = 0 ; k_row < conn.overlap ; ++k_row) {
            for (int k_col = 0 ; k_col < conn.overlap ; ++k_col) {
                int s_index = ((s_row+k_row) * conn.from_columns) + (s_col+k_col);

                // Row of matrix is the kernel index * row size (see above)
                int weight_offset = ((k_row*conn.overlap) + k_col) * kernel_row_size;
                sum += func(outputs[s_index], args...) *
                    weights[weight_offset + weight_col];
            }
        }
        inputs[d_index] = calc(conn.opcode, inputs[d_index], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_vector(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < conn.from_size) {
        inputs[index] = calc(conn.opcode, inputs[index],
            func(outputs[index], args...) * weights[index]);
    }
}
#else
template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    // IMPORTANT:
    // Serial implementation is faster if matrix is interpreted in a transposed
    //    fashion compared to parallel.  In this loop, row is the destination,
    //    column is the source.  In this way, inputs to one neuron are
    //    contiguous in memory.
    for (int row = 0 ; row < conn.to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < conn.from_size ; ++col) {
            sum += func(outputs[col], args...) *
                weights[row*conn.from_size + col];
        }
        inputs[row] = calc(conn.opcode, inputs[row], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_divergent(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int kernel_size = conn.overlap * conn.overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < conn.to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < conn.to_columns ; ++d_col) {
            int d_index = d_row*conn.to_columns + d_col;
            float sum = 0.0;

            // Determine range of source neurons for divergent kernel
            int start_s_row = d_row / conn.overlap;
            int start_s_col = d_col / conn.overlap ;
            int end_s_row = (d_row + conn.stride) / conn.overlap;
            int end_s_col = (d_col + conn.stride) / conn.overlap ;

            // Iterate over relevant source neurons...
            for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
                for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                    int s_index = (s_row * conn.from_columns) + s_col;
                    int k_row = (d_row + ((conn.overlap - conn.stride) * s_row) % conn.overlap);
                    int k_col = (d_col + ((conn.overlap - conn.stride) * s_col) % conn.overlap);

                    // Row of matrix is either the first column (convolutional)
                    //   or the index of the source neuron otherwise
                    int weight_offset = (conn.convolutional) ? 0 : s_index * kernel_size;
                    // Column of matrix is the kernel index
                    int k_index = (k_row * conn.overlap) + k_col;

                    sum += func(outputs[s_index], args...) *
                        weights[weight_offset + k_index];
                }
            }
            inputs[d_index] = calc(conn.opcode, inputs[d_index], sum);
        }
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_convergent(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int kernel_size = conn.overlap * conn.overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < conn.to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < conn.to_columns ; ++d_col) {
            int d_index = d_row*conn.to_columns + d_col;
            float sum = 0.0;

            // Determine starting row and column for source neurons
            int s_row = d_row * conn.stride;
            int s_col = d_col * conn.stride;

            // Row of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_offset = (conn.convolutional) ? 0 : d_index * kernel_size;

            // Run the kernel
            for (int k_row = 0 ; k_row < conn.overlap ; ++k_row) {
                for (int k_col = 0 ; k_col < conn.overlap ; ++k_col) {
                    int s_index = ((s_row+k_row) * conn.from_columns) + (s_col+k_col);

                    // Column of matrix is the kernel index
                    int weight_col = (k_row*conn.overlap) + k_col;
                    sum += func(outputs[s_index], args...) *
                        weights[weight_offset + weight_col];
                }
            }
            inputs[d_index] = calc(conn.opcode, inputs[d_index], sum);
        }
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_vector(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    for (int index = 0 ; index < conn.from_size ; ++index) {
        inputs[index] = calc(conn.opcode, inputs[index],
            func(outputs[index], args...) * weights[index]);
    }
}
#endif

#endif
