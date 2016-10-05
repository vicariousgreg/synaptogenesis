#include <math.h>

#include "driver/rate_encoding_driver.h"

/*****************************************************************************/
/************************* GENERIC IMPLEMENTATIONS ***************************/
/*****************************************************************************/

void RateEncodingDriver::step_connection_fully_connected(Connection *conn) {
#ifdef PARALLEL
    int blocks = calc_blocks(conn->to_layer->size);
    calc_matrix<<<blocks, THREADS>>>(
        (float*)this->re_state->device_output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->device_input + conn->to_layer->index,
        conn->from_layer->size,
        conn->to_layer->size,
        conn->opcode);
    cudaCheckError("Failed to calculate connection activation!");
#else
    calc_matrix(
        (float*)this->re_state->output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->input + conn->to_layer->index,
        conn->from_layer->size,
        conn->to_layer->size,
        conn->opcode);
#endif
}

void RateEncodingDriver::step_connection_one_to_one(Connection *conn) {
#ifdef PARALLEL
    int blocks = calc_blocks(conn->to_layer->size);
    activate_vector<<<blocks, THREADS>>>(
        (float*)this->re_state->device_output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->device_input + conn->to_layer->index,
        conn->to_layer->size,
        conn->opcode);
    cudaCheckError("Failed to calculate connection activation!");
#else

    activate_vector(
        (float*)this->re_state->output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->input + conn->to_layer->index,
        conn->to_layer->size,
        conn->opcode);
#endif
}

void RateEncodingDriver::step_connection_divergent(Connection *conn, bool convolutional) {
#ifdef PARALLEL
    dim3 blocks_per_grid(
        calc_blocks(conn->to_layer->rows, 1),
        calc_blocks(conn->to_layer->columns, 128));
    dim3 threads_per_block(1, 128);
    calc_matrix_divergent<<<blocks_per_grid, threads_per_block>>>(
        (float*)this->re_state->device_output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->device_input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional);
    cudaCheckError("Failed to calculate connection activation!");
#else
    calc_matrix_divergent(
        (float*)this->re_state->output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional);
#endif
}

void RateEncodingDriver::step_connection_convergent(Connection *conn, bool convolutional) {
#ifdef PARALLEL
    dim3 blocks_per_grid(
        calc_blocks(conn->to_layer->rows, 1),
        calc_blocks(conn->to_layer->columns, 128));
    dim3 threads_per_block(1, 128);
    calc_matrix_convergent<<<blocks_per_grid, threads_per_block>>>(
        (float*)this->re_state->device_output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->device_input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional);
    cudaCheckError("Failed to calculate connection activation!");
#else
    calc_matrix_convergent(
        (float*)this->re_state->output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional);
#endif
}

void RateEncodingDriver::step_output() {
    RateEncodingState* state = (RateEncodingState*) this->state;

#ifdef PARALLEL
    int blocks = calc_blocks(this->model->num_neurons);
    activation_function<<<blocks, THREADS>>>(
        (float*)state->device_output,
        state->device_input,
        state->device_neuron_parameters,
        this->model->num_neurons);
    cudaCheckError("Failed to update neuron output!");
#else
    activation_function(
        (float*)state->output,
        state->input,
        state->neuron_parameters,
        this->model->num_neurons);
#endif
}

void RateEncodingDriver::step_weights() {
#ifdef PARALLEL
    cudaCheckError("Failed to update connection weights!");
#endif
}


#ifdef PARALLEL
/*****************************************************************************/
/************************ PARALLEL IMPLEMENTATIONS ***************************/
/*****************************************************************************/

KERNEL void calc_matrix(float* outputs, float* weights,
        float* inputs, int from_size, int to_size, Opcode opcode) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < to_size) {
        float sum = 0;
        for (int row = 0 ; row < from_size ; ++row) {
            sum += outputs[row] * weights[row * to_size + col];
        }
        inputs[col] = calc(opcode, inputs[col], sum);
    }
}

KERNEL void calc_matrix_divergent(float* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*to_columns + d_col;

    if (d_row < to_rows and d_col < to_columns) {
        float sum = 0.0;

        // Determine range of source neurons for divergent kernel
        int start_s_row = d_row / overlap;
        int start_s_col = d_col / overlap ;
        int end_s_row = (d_row + stride) / overlap;
        int end_s_col = (d_col + stride) / overlap ;

        // Kernels are organized into columns
        // One kernel per source neuron
        //   Unless convolutional (shared kernel)
        int kernel_size = overlap * overlap;
        int kernel_row_size = (convolutional) ? 1 : from_rows * from_columns;

        // Iterate over relevant source neurons...
        for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
            for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                int s_index = (s_row * from_columns) + s_col;
                int k_row = (d_row + ((overlap - stride) * s_row) % overlap);
                int k_col = (d_col + ((overlap - stride) * s_col) % overlap);

                // Row of matrix is the kernel index * row size (see above)
                int weight_offset = ((k_row * overlap) + k_col) * kernel_row_size;
                // Column of matrix is either the first column (convolutional)
                //   or the index of the source neuron otherwise
                int weight_col = (convolutional) ? 0 : s_index;

                sum += outputs[s_index] *
                    weights[weight_offset + weight_col];
            }
        }
        inputs[d_index] = calc(opcode, inputs[d_index], sum);
    }
}

KERNEL void calc_matrix_convergent(float* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*to_columns + d_col;

    if (d_row < to_rows and d_col < to_columns) {
        float sum = 0.0;
        int s_row = d_row * stride;
        int s_col = d_col * stride;

        // Kernels are organized into columns
        // One kernel per destination neuron
        //   Unless convolutional (shared kernel)
        int kernel_size = overlap * overlap;
        int kernel_row_size = (convolutional) ? 1 : to_rows * to_columns;

        // Column of matrix is either the first column (convolutional)
        //   or the index of the destination neuron otherwise
        int weight_col = (convolutional) ? 0 : d_index;

        // Run the kernel
        for (int k_row = 0 ; k_row < overlap ; ++k_row) {
            for (int k_col = 0 ; k_col < overlap ; ++k_col) {
                int s_index = ((s_row+k_row) * from_columns) + (s_col+k_col);

                // Row of matrix is the kernel index * row size (see above)
                int weight_offset = ((k_row*overlap) + k_col) * kernel_row_size;
                sum += outputs[s_index] *
                    weights[weight_offset + weight_col];
            }
        }
        outputs[d_index] = calc(opcode, outputs[d_index], sum);
    }
}

KERNEL void activate_vector(float* outputs, float* weights,
                    float* inputs, int size, Opcode opcode) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        inputs[index] = calc(opcode, inputs[index],
            outputs[index] * weights[index]);
    }
}

KERNEL void activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < num_neurons and inputs[nid] > 0.0) {
        outputs[nid] = tanh(0.1*inputs[nid]);
    }
}

#else
/*****************************************************************************/
/************************** SERIAL IMPLEMENTATIONS ***************************/
/*****************************************************************************/

void calc_matrix(float* outputs, float* weights, float* inputs,
                        int from_size, int to_size, Opcode opcode) {
    // IMPORTANT:
    // Serial implementation is faster if matrix is interpreted in a transposed
    //    fashion compared to parallel.  In this loop, row is the destination,
    //    column is the source.  In this way, inputs to one neuron are
    //    contiguous in memory.
    for (int row = 0 ; row < to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < from_size ; ++col) {
            sum += outputs[col] *
                weights[row*from_size + col];
        }
        inputs[row] = calc(opcode, inputs[row], sum);
    }
}

void calc_matrix_divergent(float* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional) {
    int kernel_size = overlap * overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) {
            int d_index = d_row*to_columns + d_col;
            float sum = 0.0;

            // Determine range of source neurons for divergent kernel
            int start_s_row = d_row / overlap;
            int start_s_col = d_col / overlap ;
            int end_s_row = (d_row + stride) / overlap;
            int end_s_col = (d_col + stride) / overlap ;

            // Iterate over relevant source neurons...
            for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
                for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                    int s_index = (s_row * from_columns) + s_col;
                    int k_row = (d_row + ((overlap - stride) * s_row) % overlap);
                    int k_col = (d_col + ((overlap - stride) * s_col) % overlap);

                    // Row of matrix is either the first column (convolutional)
                    //   or the index of the source neuron otherwise
                    int weight_offset = (convolutional) ? 0 : s_index * kernel_size;
                    // Column of matrix is the kernel index
                    int k_index = (k_row * overlap) + k_col;

                    sum += outputs[s_index] *
                        weights[weight_offset + k_index];
                }
            }
            inputs[d_index] = calc(opcode, inputs[d_index], sum);
        }
    }
}

void calc_matrix_convergent(float* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional) {
    int kernel_size = overlap * overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) {
            int d_index = d_row*to_columns + d_col;
            float sum = 0.0;

            // Determine starting row and column for source neurons
            int s_row = d_row * stride;
            int s_col = d_col * stride;

            // Row of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_offset = (convolutional) ? 0 : d_index * kernel_size;

            // Run the kernel
            for (int k_row = 0 ; k_row < overlap ; ++k_row) {
                for (int k_col = 0 ; k_col < overlap ; ++k_col) {
                    int s_index = ((s_row+k_row) * from_columns) + (s_col+k_col);

                    // Column of matrix is the kernel index
                    int weight_col = (k_row*overlap) + k_col;
                    sum += outputs[s_index] *
                        weights[weight_offset + weight_col];
                }
            }
            inputs[d_index] = calc(opcode, inputs[d_index], sum);
        }
    }
}

void activate_vector(float* outputs, float* weights, float* inputs,
                                        int size, Opcode opcode) {
    for (int index = 0 ; index < size ; ++index) {
        inputs[index] = calc(opcode, inputs[index],
            outputs[index] * weights[index]);
    }
}

void activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons) {
    RateEncodingParameters *params;

    for (int nid = 0 ; nid < num_neurons; ++nid) {
        if (inputs[nid] > 0.0) {
            outputs[nid] = tanh(0.1*inputs[nid]);
        }
    }
}

#endif
