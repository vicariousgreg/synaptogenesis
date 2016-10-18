#ifndef izhikevich_driver_h
#define izhikevich_driver_h

#include <iostream>
#include <vector>

#include "driver/driver.h"
#include "state/izhikevich_state.h"
#include "parallel.h"

class IzhikevichDriver : public Driver {
    public:
        IzhikevichDriver(Model *model);

        OutputType get_output_type() { return INT; }
        int get_timesteps_per_output() { return 32; }

        void step_connections();
        void step_state();
        void step_weights();

        IzhikevichState *iz_state;
        float(*calc_input_ptr)(Output, int);
};

/* Parallel implementation of Izhikevich voltage update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
GLOBAL void izhikevich(float* voltages, float*recoveries, float* currents,
                IzhikevichParameters* neuron_params, int num_neurons);

/* Parallel implementation of spike update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
GLOBAL void calc_spikes(int* spikes, float* voltages, float* recoveries,
                 IzhikevichParameters* neuron_params, int num_neurons);

#endif
