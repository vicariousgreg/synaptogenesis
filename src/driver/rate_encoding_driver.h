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
        int get_timesteps_per_output() { return 1; }

        void update_connection(Instruction *inst);
        void update_state(int start_index, int count);
        void update_weights(Instruction *inst);

        RateEncodingState *re_state;
        float(*calc_input_ptr)(Output);
};

GLOBAL void activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params,
                int start_index, int count);
GLOBAL void shift_output(float* outputs,
                int start_index, int count, int num_neurons);

#endif
