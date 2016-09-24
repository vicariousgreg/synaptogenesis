#ifndef operations_h
#define operations_h

#include "neuron_parameters.h"
#include "constants.h"

#ifdef PARALLEL

/* Parallel implementation of multiplication function for activation of
 *   neural connections.  Parameters are pointers to locations in various
 *   value arrays, which allows for optimzation of memory access.  In addition,
 *   the size of the matrix is provided via |from_size| and to_size|.
 *
 * The parallel kernel calculates the input to one neuron, the weights of which
 *   are located in a column of the matrix.  This is efficient because threads
 *   running simultaneously will access sequential data from one row.
 */
__global__ void mult(int sign, int* spikes, float* weights, float* currents,
          int from_size, int to_size);

/* Parallel implementation of Izhikevich voltage update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
__global__ void izhikevich(float* voltages, float*recoveries, float* currents,
                NeuronParameters* neuron_params, int num_neurons);

/* Parallel implementation of spike update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
__global__ void calc_spikes(int* spikes, float* voltages, float* recoveries,
                 NeuronParameters* neuron_params, int num_neurons);

#else

/* Serial implementation of multiplication function for activation of
 *   neural connections */
void mult(int sign, int* spikes, float* weights, float* currents,
          int from_size, int to_size);

/* Serial implementation of Izhikevich voltage update function */
void izhikevich(float* voltages, float*recoveries, float* currents,
                NeuronParameters* neuron_params, int num_neurons);

/* Serial implementation of spike update function */
void calc_spikes(int* spikes, float* voltages, float* recoveries,
                 NeuronParameters* neuron_params, int num_neurons);

#endif

#endif
