#ifndef attribute_kernel_h
#define attribute_kernel_h

#include <string>

#include "util/parallel.h"

class Attributes;

/* Typedef for attribute kernel functions */
typedef void(*ATTRIBUTE_KERNEL)(Attributes*, int, int);

/* Attribute kernels are responsible for updating neuron attributes */
void get_attribute_kernel(ATTRIBUTE_KERNEL *dest, std::string engine_name);

/* Izhikevich voltage update and spike calculation */
GLOBAL void iz_update_attributes(Attributes *att, int start_index, int count);

/* Shifts output and computes most recent output
 * using positive tanh Activation function */
GLOBAL void re_update_attributes(Attributes *att, int start_index, int count);

/* Hodgkin-Huxley voltage update and spike calculation */
GLOBAL void hh_update_attributes(Attributes *att, int start_index, int count);

#endif
