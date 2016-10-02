#ifndef izhikevich_driver_h
#define izhikevich_driver_h

#include <iostream>

#include "driver.h"
#include "izhikevich_state.h"
#include "parallel.h"

class IzhikevichDriver : public Driver {
    public:
        IzhikevichDriver () {
            this->iz_state = new IzhikevichState();
            this->state = this->iz_state;
        }

        void step_connection_fully_connected(Connection *conn);
        void step_connection_one_to_one(Connection *conn);
        void step_connection_divergent(Connection *conn);
        void step_connection_convergent(Connection *conn, bool convolutional);
        void step_output();
        void step_weights();

        IzhikevichState *iz_state;
};


/* Generic versions to obfuscate preprocessor directives. */
void iz_update_currents(Connection *conn, float* mData, int* spikes,
                     float* currents, int num_neurons);

#ifdef PARALLEL

/* Parallel implementation of activation function for activation of
 *   neural connections.  Parameters are pointers to locations in various
 *   value arrays, which allows for optimzation of memory access.  In addition,
 *   the size of the matrix is provided via |from_size| and to_size|.
 */
 
/* This parallel kernel calculates the input to one neuron, the weights of which
 *   are located in a column of the matrix.  This is efficient because threads
 *   running calc_matrix will access sequential data from one row.
 */
__global__ void parallel_calc_matrix(int* spikes, float* weights,
        float* currents, int from_size, int to_size, int mask, Opcode opcode,
        int overlap, int stride);

__global__ void parallel_calc_matrix_divergent(int* spikes, float* weights,
        float* currents, int from_rows, int from_coluns, int to_rows, int to_columns,
        int mask, Opcode opcode, int overlap, int stride);

__global__ void parallel_calc_matrix_convergent(int* spikes, float* weights,
        float* currents, int from_rows, int from_columns, int to_rows, int to_columns,
        int mask, Opcode opcode, int overlap, int stride, bool convolutional);

/* This parallel kernel calculates the input to one neuron, which only ahs one
 *   input weight.  Weight vectors represent one-to-one neural connections.
 */
__global__ void parallel_activate_vector(int* spikes, float* weights,
                    float* currents, int size, int mask, Opcode opcode);

/* Parallel implementation of Izhikevich voltage update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
__global__ void parallel_izhikevich(float* voltages, float*recoveries, float* currents,
                IzhikevichParameters* neuron_params, int num_neurons);

/* Parallel implementation of spike update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
__global__ void parallel_calc_spikes(int* spikes, float* voltages, float* recoveries,
                 IzhikevichParameters* neuron_params, int num_neurons);

#else

/* Serial implementation of calc_matrix functions for activation of
 *   neural connections */
void serial_calc_matrix(int* spikes, float* weights, float* currents,
                        int from_size, int to_size, int mask, Opcode opcode);

void serial_calc_matrix_divergent(int* spikes, float* weights,
        float* currents, int from_rows, int from_columns, int to_rows, int to_columns,
        int mask, Opcode opcode, int overlap, int stride);

void serial_calc_matrix_convergent(int* spikes, float* weights,
        float* currents, int from_rows, int from_columns, int to_rows, int to_columns,
        int mask, Opcode opcode, int overlap, int stride, bool convolutional);

/* Serial implementation of activate_vector function for activation of
 *   neural connections */
void serial_activate_vector(int* spikes, float* weights, float* currents,
                                        int size, int mask, Opcode opcode);

/* Serial implementation of Izhikevich voltage update function */
void serial_izhikevich(float* voltages, float* recoveries, float* currents,
                IzhikevichParameters* neuron_params, int num_neurons);

/* Serial implementation of spike update function */
void serial_calc_spikes(int* spikes, float* voltages, float* recoveries,
                 IzhikevichParameters* neuron_params, int num_neurons);

#endif

#endif
