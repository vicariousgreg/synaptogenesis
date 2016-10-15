#ifndef izhikevich_state_h
#define izhikevich_state_h

#include "state/state.h"

/* Neuron parameters class.
 * Contains a,b,c,d parameters for Izhikevich model */
class IzhikevichParameters {
    public:
        IzhikevichParameters(float a, float b, float c, float d) :
                a(a), b(b), c(c), d(d) {}
        float a, b, c, d;
};

class IzhikevichState : public State {
    public:
        void build(Model* model, int output_size);

        //////////////////////
        /// MODEL SPECIFIC ///
        //////////////////////
        // GETTERS
        /* If parallel, these will copy data from the device */
        float* get_voltage();
        float* get_recovery();

    private:
        friend class IzhikevichDriver;

        // Neuron States
        float *voltage;
        float *recovery;

        // Neuron Spikes
        int* spikes;

        // Neuron parameters
        IzhikevichParameters* neuron_parameters;
};

#endif
