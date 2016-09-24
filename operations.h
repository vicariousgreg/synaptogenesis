#ifndef operations_h
#define operations_h

#include <stdint.h>

#include "neuron_parameters.h"
#include "parallel.h"

class WeightMatrix;

/* Synaptic operation opcode */
enum OPCODE {
    ADD,
    SUB,
    MULT,
    DIV
};

/* Generic versions to obfuscate preprocessor directives. */
void update_currents(WeightMatrix &conn, int* spikes,
                    float* currents, int num_neurons);

void update_voltages(float* voltages, float*recoveries, float* currents,
                NeuronParameters* neuron_params, int num_neurons);

void update_spikes(int* spikes, float* voltages, float* recoveries,
                 NeuronParameters* neuron_params, int num_neurons);

void update_weights();

#ifdef PARALLEL

__device__ float calc(OPCODE opcode, float current, float sum);

/* Parallel implementation of activation function for activation of
 *   neural connections.  Parameters are pointers to locations in various
 *   value arrays, which allows for optimzation of memory access.  In addition,
 *   the size of the matrix is provided via |from_size| and to_size|.
 */
 
/* This parallel kernel calculates the input to one neuron, the weights of which
 *   are located in a column of the matrix.  This is efficient because threads
 *   running siactivate_matrix will access sequential data from one row.
 */
__global__ void parallel_activate_matrix(int* spikes, float* weights,
        float* currents, int from_size, int to_size, int mask, OPCODE opcode);

/* This parallel kernel calculates the input to one neuron, which only ahs one
 *   input weight.  Weight vectors represent one-to-one neural connections.
 */
__global__ void parallel_activate_vector(int* spikes, float* weights,
                    float* currents, int size, int mask, OPCODE opcode);

/* Parallel implementation of Izhikevich voltage update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
__global__ void parallel_izhikevich(float* voltages, float*recoveries, float* currents,
                NeuronParameters* neuron_params, int num_neurons);

/* Parallel implementation of spike update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
__global__ void parallel_calc_spikes(int* spikes, float* voltages, float* recoveries,
                 NeuronParameters* neuron_params, int num_neurons);

#else

float calc(OPCODE opcode, float current, float sum);

/* Serial implementation of activate_matrix function for activation of
 *   neural connections */
void serial_activate_matrix(int* spikes, float* weights, float* currents,
                        int from_size, int to_size, int mask, OPCODE opcode);

/* Serial implementation of activate_vector function for activation of
 *   neural connections */
void serial_activate_vector(int* spikes, float* weights, float* currents,
                                        int size, int mask, OPCODE opcode);

/* Serial implementation of Izhikevich voltage update function */
void serial_izhikevich(float* voltages, float*recoveries, float* currents,
                NeuronParameters* neuron_params, int num_neurons);

/* Serial implementation of spike update function */
void serial_calc_spikes(int* spikes, float* voltages, float* recoveries,
                 NeuronParameters* neuron_params, int num_neurons);

#endif

#endif
