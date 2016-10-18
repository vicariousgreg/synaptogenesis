#ifndef izhikevich_driver_h
#define izhikevich_driver_h

#include <iostream>
#include <vector>

#include "driver/driver.h"
#include "driver/instruction.h"
#include "state/izhikevich_state.h"
#include "parallel.h"

class IzhikevichDriver : public Driver {
    public:
        IzhikevichDriver();

        void build_instructions();

        void step_connections();
        void step_state();
        void step_weights();

        int get_output_size() { return sizeof(int); }

        IzhikevichState *iz_state;
        float(*calc_input_ptr)(int, int);

        std::vector<Instruction<int>* > instructions;
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
