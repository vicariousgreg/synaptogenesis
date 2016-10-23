#include <math.h>

#include "driver/rate_encoding_driver.h"
#include "state/state.h"
#include "parallel.h"

/* Activation function */
GLOBAL void activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params,
                int start_index, int count);

/* Output shifter */
GLOBAL void shift_output(float* outputs,
                int start_index, int count, int num_neurons);

DEVICE float re_calc_input(Output output) { return output.f; }
DEVICE float (*re_calc_input_ptr)(Output) = re_calc_input;

RateEncodingDriver::RateEncodingDriver(Model *model) : Driver() {
    this->re_attributes = new RateEncodingAttributes(model);
    this->state = new State(model, this->re_attributes, 1);
#ifdef PARALLEL
    cudaMemcpyFromSymbol(&this->calc_input_ptr, re_calc_input_ptr, sizeof(void *));
#else
    this->calc_input_ptr = re_calc_input_ptr;
#endif
}

void RateEncodingDriver::update_connection(Instruction *inst) {
    step<>(inst, this->calc_input_ptr);
}

void RateEncodingDriver::update_state(int start_index, int count) {
#ifdef PARALLEL
    int threads = calc_threads(count);
    int blocks = calc_blocks(count);
    shift_output<<<blocks, threads, 0, *this->curr_stream>>>(
#else
    shift_output(
#endif
        (float*)re_attributes->output, start_index, count, re_attributes->total_neurons);
#ifdef PARALLEL
    cudaCheckError("Failed to update neuron output!");

    activation_function<<<blocks, threads, 0, *this->curr_stream>>>(
#else
    activation_function(
#endif
        (float*)re_attributes->recent_output,
        re_attributes->input,
        re_attributes->neuron_parameters,
        start_index, count);
#ifdef PARALLEL
    cudaCheckError("Failed to update neuron output!");
#endif
}

void RateEncodingDriver::update_weights(Instruction *inst) {
#ifdef PARALLEL
    //cudaCheckError("Failed to update connection weights!");
#endif
}


GLOBAL void shift_output(float* outputs,
        int start_index, int count, int num_neurons) {
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
#else
    for (int nid = start_index ; nid < start_index+count; ++nid) {
#endif
        float curr_value, next_value = outputs[nid];
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++ index) {
            curr_value = next_value;
            next_value = outputs[num_neurons * (index + 1) + nid];
            outputs[num_neurons*index + nid] = next_value;
        }
        outputs[num_neurons*index + nid] = next_value;
    }
}

GLOBAL void activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params,
                int start_index, int count) {
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        nid += start_index;
#else
    for (int nid = start_index ; nid < start_index+count; ++nid) {
#endif
        float input = inputs[nid];
        outputs[nid] = (input > 0.0) ? tanh(0.1*input) : 0.0;
    }
}
