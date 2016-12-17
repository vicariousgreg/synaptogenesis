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
        IzhikevichState(Model* model);
        ~IzhikevichState();

        float* get_voltage() { return this->voltage; }
        float* get_recovery() { return this->recovery; }
        int* get_spikes() { return this->spikes; }
        IzhikevichParameters* get_parameters() {
            return this->neuron_parameters;
        }

        // Neuron Attributes
        float *voltage;
        float *recovery;

        // Neuron Current (copy of input)
        float* current;

        // Neuron Spikes (copy of output)
        int* spikes;

        // Neuron parameters
        IzhikevichParameters* neuron_parameters;
};

#endif
