#include <math.h>

#include "driver/rate_encoding_driver.h"

DEVICE float re_calc_input(float output) {
    return output;
}

DEVICE float (*re_calc_input_ptr)(float) = re_calc_input;

RateEncodingDriver::RateEncodingDriver () {
    this->re_state = new RateEncodingState();
    this->state = this->re_state;

#ifdef PARALLEL
    cudaMemcpyFromSymbol(&this->calc_input_ptr, re_calc_input_ptr, sizeof(void *));
#else
    this->calc_input_ptr = re_calc_input_ptr;
#endif
}

/*****************************************************************************/
/************************* GENERIC IMPLEMENTATIONS ***************************/
/*****************************************************************************/

void RateEncodingDriver::step_connection_fully_connected(Connection *conn) {
#ifdef PARALLEL
    int blocks = calc_blocks(conn->to_layer->size);
    calc_matrix<float><<<blocks, THREADS>>>(
        this->calc_input_ptr,
        (float*)this->re_state->device_output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->device_input + conn->to_layer->index,
        conn->from_layer->size,
        conn->to_layer->size,
        conn->opcode);
    cudaCheckError("Failed to calculate connection activation!");
#else
    calc_matrix<float>(
        this->calc_input_ptr,
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
    activate_vector<float><<<blocks, THREADS>>>(
        this->calc_input_ptr,
        (float*)this->re_state->device_output + conn->from_layer->index,
        this->re_state->get_matrix(conn->id),
        this->re_state->device_input + conn->to_layer->index,
        conn->to_layer->size,
        conn->opcode);
    cudaCheckError("Failed to calculate connection activation!");
#else

    activate_vector<float>(
        this->calc_input_ptr,
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
    calc_matrix_divergent<float><<<blocks_per_grid, threads_per_block>>>(
        this->calc_input_ptr,
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
    calc_matrix_divergent<float>(
        this->calc_input_ptr,
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
    calc_matrix_convergent<float><<<blocks_per_grid, threads_per_block>>>(
        this->calc_input_ptr,
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
    calc_matrix_convergent<float>(
        this->calc_input_ptr,
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

GLOBAL void activation_function(float* outputs, float* inputs,
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
