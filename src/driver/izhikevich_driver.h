#ifndef izhikevich_driver_h
#define izhikevich_driver_h

#include <iostream>
#include <vector>

#include "driver/driver.h"
#include "state/izhikevich_attributes.h"

class IzhikevichDriver : public Driver {
    public:
        IzhikevichDriver(Model *model);

        int get_timesteps_per_output() { return 32; }

        void update_connection(Instruction *inst);
        void update_state(int start_index, int count);
        void update_weights(Instruction *inst);

        IzhikevichAttributes *iz_attributes;
        float(*calc_input_ptr)(Output, int);
};

#endif
