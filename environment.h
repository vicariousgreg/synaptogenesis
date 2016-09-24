#ifndef environment_h
#define environment_h

#include "state.h"
#include "model.h"

class Environment {
    public:
        Environment (Model model);

        virtual ~Environment () {}

        /* Builds the environment.
         * Allocates memory for various arrays of data.
         * Calls build() on all weight matrices.
         * Returns whether allocation was successful.
         */
        void build();

        /* Injects current from the given pointer to the given layer */
        void inject_current(int layer_id, float* input);

        /* Injects randomized current to the given layer.
         * Input is bounded by |max| */
        void inject_random_current(int layer_id, float max);

        /* Zeroes current to the given layer */
        void clear_current(int layer_id);

        /* Returns a pointer to an array containing the most recent integer
         *   of neuron spikes.
         * If parallel, this will copy the values from the device. */
        int* get_spikes();

        /* Returns a pointer to an array containing the neuron currents.
         * If parallel, this will copy the values from the device. */
        float* get_current();

        /**********************************************************************
         *********************** TIMESTEP DYNAMICS ****************************
         **********************************************************************/

        /* Cycles the environment:
         * 1. Activate neural connections, which updates currents.
         * 2. Update neuron voltages from currents.
         * 3. Timestep the spikes.
         * 4. Update connection weights for plastic matrices */
        void cycle();

        /* Activates neural connections, triggering updates of currents */
        void activate();

        /* Updates neuron voltages from the currents using the Izhikevich model */
        void update_voltages();

        /* Timesteps the spikes, shifting spike bit vectors */
        void timestep();

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        void update_weights();

        // Environment state
        State state;

        // Network model
        Model model;
};

#endif
