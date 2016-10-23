#ifndef rate_encoding_driver_h
#define rate_encoding_driver_h

#include <iostream>

#include "driver/driver.h"
#include "state/rate_encoding_attributes.h"

class RateEncodingDriver : public Driver {
    public:
        RateEncodingDriver(Model *model);

        OutputType get_output_type() { return FLOAT; }
        int get_timesteps_per_output() { return 1; }

        void update_connection(Instruction *inst);
        void update_state(int start_index, int count);
        void update_weights(Instruction *inst);

        RateEncodingAttributes *re_attributes;
        float(*calc_input_ptr)(Output);
};

#endif
