#include <math.h>

#include "engine/kernel/rate_encoding_kernel.h"

GLOBAL void re_update_output(RateEncodingAttributes *att,
        int start_index, int count, int num_neurons) {
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
#else
    for (int nid = start_index ; nid < start_index+count; ++nid) {
#endif
        float curr_value, next_value = att->output[nid];
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++index) {
            curr_value = next_value;
            next_value = att->output[num_neurons * (index + 1) + nid];
            att->output[num_neurons*index + nid] = next_value;
        }
        float input = att->input[nid];
        att->output[num_neurons*index + nid] =
            (input > 0.0) ? tanh(0.01*input) : 0.0;
    }
}
