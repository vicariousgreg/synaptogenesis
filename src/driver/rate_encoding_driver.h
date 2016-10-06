#ifndef rate_encoding_odriver_hperations_h
#define rate_encoding_driver_h

#include <iostream>

#include "driver/driver.h"
#include "state/rate_encoding_state.h"
#include "parallel.h"

class RateEncodingDriver : public Driver {
    public:
        RateEncodingDriver ();

        void step_connection(Connection *conn);
        void step_output();
        void step_weights();

        RateEncodingState *re_state;
        float(*calc_input_ptr)(float);
};

GLOBAL void activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons);

#endif
