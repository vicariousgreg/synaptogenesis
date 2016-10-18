#ifndef rate_encoding_odriver_hperations_h
#define rate_encoding_driver_h

#include <iostream>

#include "driver/driver.h"
#include "state/rate_encoding_state.h"
#include "parallel.h"

class RateEncodingDriver : public Driver {
    public:
        RateEncodingDriver(Model *model);

        OutputType get_output_type() { return FLOAT; }

        void step_connections();
        void step_state();
        void step_weights();

        RateEncodingState *re_state;
        float(*calc_input_ptr)(float);

        std::vector<Instruction<float>* > instructions;
};

GLOBAL void activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons);
GLOBAL void shift_output(float* outputs, int num_neurons);

#endif
