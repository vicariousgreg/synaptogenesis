#ifndef rate_encoding_odriver_hperations_h
#define rate_encoding_driver_h

#include <iostream>

#include "driver/driver.h"
#include "state/rate_encoding_state.h"
#include "parallel.h"


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

KERNEL void calc_matrix(float* outputs, float* weights,
        float* inputs, int from_size, int to_size, Opcode opcode);

KERNEL void calc_matrix_divergent(float* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional);

KERNEL void calc_matrix_convergent(float* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional);

KERNEL void activate_vector(float* outputs, float* weights,
                    float* inputs, int size, Opcode opcode);

KERNEL void activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons);

#endif
