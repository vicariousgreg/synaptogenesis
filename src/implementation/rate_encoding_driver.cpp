#include <math.h>

#include "rate_encoding_driver.h"

/*****************************************************************************/
/************************* GENERIC IMPLEMENTATIONS ***************************/
/*****************************************************************************/

void RateEncodingDriver::step_connection_fully_connected(Connection *conn) {
#ifdef PARALLEL
    int blocks = calc_blocks(conn->to_layer->size);
    parallel_calc_matrix<<<blocks, THREADS>>>(
        (float*)this->re_state->device_output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->device_input + conn->to_layer->index,
        conn->from_layer->size,
        conn->to_layer->size,
        conn->opcode);
    cudaCheckError("Failed to calculate connection activation!");
#else
    serial_calc_matrix(
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
    parallel_activate_vector<<<blocks, THREADS>>>(
        (float*)this->re_state->device_output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->device_input + conn->to_layer->index,
        conn->to_layer->size,
        conn->opcode);
    cudaCheckError("Failed to calculate connection activation!");
#else

    serial_activate_vector(
        (float*)this->re_state->output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->input + conn->to_layer->index,
        conn->to_layer->size,
        conn->opcode);
#endif
}

void RateEncodingDriver::step_connection_divergent(Connection *conn) {
    throw "Divergent connection unimplemented!";
}

void RateEncodingDriver::step_connection_convergent(Connection *conn, bool convolutional) {
#ifdef PARALLEL
    dim3 blocks_per_grid(
        calc_blocks(conn->to_layer->rows, 1),
        calc_blocks(conn->to_layer->columns, 128));
    dim3 threads_per_block(1, 128);
    parallel_calc_matrix_convergent<<<blocks_per_grid, threads_per_block>>>(
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
    serial_calc_matrix_convergent(
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
    parallel_activation_function<<<blocks, THREADS>>>(
        (float*)state->device_output,
        state->device_input,
        state->device_neuron_parameters,
        this->model->num_neurons);
    cudaCheckError("Failed to update neuron output!");
#else
    serial_activation_function(
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

__global__ void parallel_calc_matrix(float* outputs, float* weights,
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

__global__ void parallel_calc_matrix_convergent(float* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*to_columns + d_col;

    if (d_row < to_rows and d_col < to_columns) {
        int kernel_size = overlap * overlap;
        int kernel_row_size = (convolutional) ? 1 : from_rows * from_columns;

        float sum = 0.0;
        int s_row = d_row * stride;
        int s_col = d_col * stride;

        // Convolutional connections share weights, and don't use an offset
        // In parallel version, matrix is transposed, so the offset is the index.
        int kernel_offset = (convolutional) ? 0 : d_index;

        // Run the kernel
        for (int k_row = 0 ; k_row < overlap ; ++k_row) {
            for (int k_col = 0 ; k_col < overlap ; ++k_col) {
                int s_index = (s_row+k_row) * from_columns + (s_col+k_col);
                int k_index = (((k_row*overlap) + k_col) * kernel_row_size) + kernel_offset;
                sum += outputs[s_index] * weights[k_index];
            }
        }
        outputs[d_index] = calc(opcode, outputs[d_index], sum);
    }
}

__global__ void parallel_activate_vector(float* outputs, float* weights,
                    float* inputs, int size, Opcode opcode) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        inputs[index] = calc(opcode, inputs[index],
            outputs[index] * weights[index]);
    }
}

__global__ void parallel_activation_function(float* outputs, float* inputs,
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

void serial_calc_matrix(float* outputs, float* weights, float* inputs,
                        int from_size, int to_size, Opcode opcode) {
    // IMPORTANT:
    // Serial implementation is faster if matrix is interpreted in a transposed
    //    fashion compared to parallel.  In this loop, row is the destination,
    //    column is the source.  In this way, inputs to one neuron are
    //    contiguous in memory.
    for (int row = 0 ; row < to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < from_size ; ++col) {
            sum += outputs[col] * weights[row*from_size + col];
        }
        inputs[row] = calc(opcode, inputs[row], sum);
    }
}

void serial_calc_matrix_convergent(float* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional) {
    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) {
            float sum = 0.0;
            int s_row = d_row * stride;
            int s_col = d_col * stride;
            int d_index = d_row*to_columns + d_col;

            // Convolutional connections share weights, and don't use an offset
            int kernel_offset = (convolutional) ? 0 : d_index * overlap * overlap;

            // Run the kernel (unshared)
            for (int k_row = 0 ; k_row < overlap ; ++k_row) {
                for (int k_col = 0 ; k_col < overlap ; ++k_col) {
                    int s_index = (s_row+k_row) * from_columns + (s_col+k_col);
                    sum += outputs[s_index] *
                        weights[kernel_offset + (k_row*overlap) + k_col];
                }
            }
            inputs[d_index] = calc(opcode, inputs[d_index], sum);
        }
    }
}

void serial_activate_vector(float* outputs, float* weights, float* inputs,
                                        int size, Opcode opcode) {
    for (int index = 0 ; index < size ; ++index) {
        inputs[index] = calc(opcode, inputs[index],
            outputs[index] * weights[index]);
    }
}

void serial_activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons) {
    RateEncodingParameters *params;

    for (int nid = 0 ; nid < num_neurons; ++nid) {
        if (inputs[nid] > 0.0) {
            outputs[nid] = tanh(0.1*inputs[nid]);
        }
    }
}

#endif
