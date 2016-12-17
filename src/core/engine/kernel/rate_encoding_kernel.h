#ifndef rate_encoding_kernel_h
#define rate_encoding_kernel_h

#include "state/rate_encoding_attributes.h"
#include "util/parallel.h"

/* Shifts output and computes most recent output
 * using positive tanh Activation function */
GLOBAL void re_update_attributes(RateEncodingAttributes *att,
    int start_index, int count, int num_neurons);

#endif
