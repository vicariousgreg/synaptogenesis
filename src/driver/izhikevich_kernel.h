#ifndef izhikevich_kernel_h
#define izhikevich_kernel_h

#include "state/izhikevich_attributes.h"
#include "parallel.h"

/* Izhikevich voltage update function */
GLOBAL void izhikevich(float* voltages, float*recoveries, float* currents,
                IzhikevichParameters* neuron_params,
                int start_index, int count);

/* Spike update function */
GLOBAL void calc_spikes(int* spikes, float* voltages, float* recoveries,
                 IzhikevichParameters* neuron_params,
                 int start_index, int count, int num_neurons);

#endif
