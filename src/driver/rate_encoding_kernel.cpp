#include <math.h>

#include "driver/rate_encoding_kernel.h"

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
