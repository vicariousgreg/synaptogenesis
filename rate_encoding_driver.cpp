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
    throw "Convergent connection unimplemented!";
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
