#include <math.h>

#include "driver/rate_encoding_driver.h"

DEVICE float re_calc_input(Output output) { return output.f; }
DEVICE float (*re_calc_input_ptr)(Output) = re_calc_input;

RateEncodingDriver::RateEncodingDriver(Model *model) {
    this->re_state = new RateEncodingState(model);
    this->state = this->re_state;
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
    RateEncodingState* state = (RateEncodingState*) this->state;

#ifdef PARALLEL
    int threads = 128;
    int blocks = calc_blocks(count, threads);
    shift_output<<<blocks, threads>>>(
#else
    shift_output(
#endif
        (float*)state->output, start_index, count, this->state->total_neurons);
#ifdef PARALLEL
    cudaCheckError("Failed to update neuron output!");

    activation_function<<<blocks, threads>>>(
#else
    activation_function(
#endif
        (float*)state->recent_output,
        state->input,
        state->neuron_parameters,
        start_index, count);
#ifdef PARALLEL
    cudaCheckError("Failed to update neuron output!");
#endif
}

void RateEncodingDriver::update_weights(Instruction *inst) {
#ifdef PARALLEL
    cudaCheckError("Failed to update connection weights!");
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
