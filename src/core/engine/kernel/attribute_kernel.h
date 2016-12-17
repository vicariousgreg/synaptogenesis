#ifndef attribute_kernel_h
#define attribute_kernel_h

#include <string>

#include "util/parallel.h"

class State;

/* Typedef for attribute kernel functions */
typedef void(*ATTRIBUTE_KERNEL)(State*, int, int, int);

/* Attribute kernels are responsible for updating neuron attributes */
void get_attribute_kernel(ATTRIBUTE_KERNEL *dest, std::string engine_name);

/* Izhikevich voltage update and spike calculation */
GLOBAL void iz_update_attributes(State *state,
    int start_index, int count, int total_neurons);

/* Shifts output and computes most recent output
 * using positive tanh Activation function */
GLOBAL void re_update_attributes(State *state,
    int start_index, int count, int total_neurons);

#endif
