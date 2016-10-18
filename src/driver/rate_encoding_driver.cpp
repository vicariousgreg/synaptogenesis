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

void RateEncodingDriver::step_connection(Instruction *inst) {
    step<>(inst, this->calc_input_ptr);
}

void RateEncodingDriver::step_state() {
    RateEncodingState* state = (RateEncodingState*) this->state;

#ifdef PARALLEL
    int threads = 128;
    int blocks = calc_blocks(this->state->num_neurons, threads);
    shift_output<<<blocks, threads>>>(
#else
    shift_output(
#endif
        (float*)state->output, this->state->num_neurons);
#ifdef PARALLEL
    cudaCheckError("Failed to update neuron output!");

    activation_function<<<blocks, threads>>>(
#else
    activation_function(
#endif
        (float*)state->recent_output,
        state->input,
        state->neuron_parameters,
        this->state->num_neurons);
#ifdef PARALLEL
    cudaCheckError("Failed to update neuron output!");
#endif
}

void RateEncodingDriver::step_weights() {
#ifdef PARALLEL
    cudaCheckError("Failed to update connection weights!");
#endif
}


GLOBAL void shift_output(float* outputs, int num_neurons) {
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < num_neurons) {
#else
    for (int nid = 0 ; nid < num_neurons; ++nid) {
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
                RateEncodingParameters* neuron_params, int num_neurons) {
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < num_neurons) {
#else
    for (int nid = 0 ; nid < num_neurons; ++nid) {
#endif
        float input = inputs[nid];
        outputs[nid] = (input > 0.0) ? tanh(0.1*input) : 0.0;
    }
}
