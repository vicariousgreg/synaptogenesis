#ifndef rate_encoding_odriver_hperations_h
#define rate_encoding_driver_h

#include <iostream>

#include "../framework/driver.h"
#include "rate_encoding_state.h"
#include "../framework/parallel.h"


class RateEncodingDriver : public Driver {
    public:
        RateEncodingDriver () {
            this->re_state = new RateEncodingState();
            this->state = this->re_state;
        }

        void step_connection_fully_connected(Connection *conn);
        void step_connection_one_to_one(Connection *conn);
        void step_connection_divergent(Connection *conn, bool convolutional);
        void step_connection_convergent(Connection *conn, bool convolutional);
        void step_output();
        void step_weights();

        RateEncodingState *re_state;
};

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
__global__ void parallel_calc_matrix(float* outputs, float* weights,
        float* inputs, int from_size, int to_size, Opcode opcode);

__global__ void parallel_calc_matrix_convergent(float* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional);

/* This parallel kernel calculates the input to one neuron, which only ahs one
 *   input weight.  Weight vectors represent one-to-one neural connections.
 */
__global__ void parallel_activate_vector(float* outputs, float* weights,
                    float* inputs, int size, Opcode opcode);

__global__ void parallel_activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons);

#else

/* Serial implementation of calc_matrix function for activation of
 *   neural connections */
void serial_calc_matrix(float* outputs, float* weights, float* inputs,
                        int from_size, int to_size, Opcode opcode);

void serial_calc_matrix_convergent(float* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional);

/* Serial implementation of activate_vector function for activation of
 *   neural connections */
void serial_activate_vector(float* outputs, float* weights, float* inputs,
                                        int size, Opcode opcode);

void serial_activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons);


#endif

#endif
