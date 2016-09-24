#ifndef state_h
#define state_h

#include "model.h"
#include "neuron_parameters.h"

class State {
    public:
        State () {}

        void build(Model model);

        // SETTERS
        /* If parallel, these will copy data to the device */
        void set_current(int offset, int size, float* input);
        void randomize_current(int offset, int size, float max);
        void clear_current(int offset, int size);

        // GETTERS
        /* If parallel, these will copy data from the device */
        int* get_spikes();
        float* get_current();
        float* get_voltage();
        float* get_recovery();

    private:
        friend class Environment;

        // Neurons
        int num_neurons;

        // Neuron States
        float *current;
        float *voltage;
        float *recovery;

        // Neuron Spikes
        int* spikes;
        // Recent points to the most recent integers for spike bit vectors
        int* recent_spikes;

        // Neuron parameters
        NeuronParameters *neuron_parameters;

#ifdef PARALLEL
        // Locations to store local copies of spikes and currents.
        // When accessed, these values will be copied here from the device.
        int* local_spikes;
        float* local_current;
#endif

};

#endif
