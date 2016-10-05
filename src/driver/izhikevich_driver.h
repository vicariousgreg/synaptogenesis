#ifndef izhikevich_driver_h
#define izhikevich_driver_h

#include <iostream>

#include "driver/driver.h"
#include "state/izhikevich_state.h"
#include "parallel.h"

class IzhikevichDriver : public Driver {
    public:
        IzhikevichDriver () {
            this->iz_state = new IzhikevichState();
            this->state = this->iz_state;
        }

        void step_connection_fully_connected(Connection *conn);
        void step_connection_one_to_one(Connection *conn);
        void step_connection_divergent(Connection *conn, bool convolutional);
        void step_connection_convergent(Connection *conn, bool convolutional);
        void step_output();
        void step_weights();

        IzhikevichState *iz_state;
};


/* Parallel implementation of activation function for activation of
 *   neural connections.  Parameters are pointers to locations in various
 *   value arrays, which allows for optimization of memory access.  In addition,
 *   the size of the matrix is provided via |from_size| and to_size|.
 */
 
/* This parallel kernel calculates the input to one neuron, the weights of which
 *   are located in a column of the matrix.  This is efficient because threads
 *   running calc_matrix will access sequential data from one row.
 */
KERNEL void calc_matrix(int* spikes, float* weights,
        float* inputs, int from_size, int to_size, int mask, Opcode opcode);

KERNEL void calc_matrix_divergent(int* spikes, float* weights,
        float* inputs, int from_rows, int from_coluns, int to_rows, int to_columns,
        int mask, Opcode opcode, int overlap, int stride, bool convolutional);

KERNEL void calc_matrix_convergent(int* spikes, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        int mask, Opcode opcode, int overlap, int stride, bool convolutional);

/* This parallel kernel calculates the input to one neuron, which only has one
 *   input weight.  Weight vectors represent one-to-one neural connections.
 */
KERNEL void activate_vector(int* spikes, float* weights,
                    float* inputs, int size, int mask, Opcode opcode);

/* Parallel implementation of Izhikevich voltage update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
KERNEL void izhikevich(float* voltages, float*recoveries, float* currents,
                IzhikevichParameters* neuron_params, int num_neurons);

/* Parallel implementation of spike update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
KERNEL void calc_spikes(int* spikes, float* voltages, float* recoveries,
                 IzhikevichParameters* neuron_params, int num_neurons);

#endif
