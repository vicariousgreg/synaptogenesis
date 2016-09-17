#ifndef operations_h
#define operations_h

#include "environment.h"

void dot(int sign, int* spikes, double* weights, double* currents,
         int from_size, int to_index);

void mult(int sign, int* spikes, double* weights, double* currents,
          int from_size, int to_size);

void izhikevich(double* voltages, double*recoveries, double* currents,
                NeuronParameters* neuron_params, int num_neurons);

void calc_spikes(int* spikes, int* ages, double* voltages, double* recoveries,
                 NeuronParameters* neuron_params, int num_neurons);

#endif
