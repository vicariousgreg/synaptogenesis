#ifndef operations_h
#define operations_h

#include "environment.h"

#ifdef parallel

__global__ void mult(int sign, int* spikes, float* weights, float* currents,
          int from_size, int to_size);

__global__ void izhikevich(float* voltages, float*recoveries, float* currents,
                NeuronParameters* neuron_params, int num_neurons);

__global__ void calc_spikes(int* spikes, int* ages, float* voltages, float* recoveries,
                 NeuronParameters* neuron_params, int num_neurons);

#else

void mult(int sign, int* spikes, float* weights, float* currents,
          int from_size, int to_size);

void izhikevich(float* voltages, float*recoveries, float* currents,
                NeuronParameters* neuron_params, int num_neurons);

void calc_spikes(int* spikes, int* ages, float* voltages, float* recoveries,
                 NeuronParameters* neuron_params, int num_neurons);

#endif

#endif
