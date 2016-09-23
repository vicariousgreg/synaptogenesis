#ifndef state_h
#define state_h

#include "neuron_parameters.h"

#define HISTORY_SIZE 8

using namespace std;

class State {
    public:
        State () {}
        virtual ~State () {}

        void build(int num_neurons, NeuronParameters* neuron_parameters);

        // SETTERS
        void set_current(int offset, int size, float* input);
        void randomize_current(int offset, int size, float max);
        void clear_current(int offset, int size);

        // GETTERS
        /* If parallel, these will copy data from the device */
        int* get_spikes();
        float* get_current();
        float* get_voltage();
        float* get_recovery();

    //private:
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

#ifdef parallel
        // Locations to store local copies of spikes and currents.
        // When accessed, these values will be copied here from the device.
        int* local_spikes;
        float* local_current;
#endif

};

#endif
