#ifndef rate_encoding_kernel_h
#define rate_encoding_kernel_h

#include "state/rate_encoding_attributes.h"
#include "util/parallel.h"

/* Activation function */
GLOBAL void activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params,
                int start_index, int count);

/* Output shifter */
GLOBAL void shift_output(float* outputs,
                int start_index, int count, int num_neurons);

#endif
