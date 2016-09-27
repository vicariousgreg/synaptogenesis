#ifndef izhikevich_state_h
#define izhikevich_state_h

#include "state.h"
#include "model.h"

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
        void build(Model* model);
        void step_input();
        void step_output();
        void step_weights();

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

#ifdef PARALLEL
        // Locations to store device copies of data.
        // When accessed, these values will be copied here from the device.
        int *device_spikes;
        float *device_voltage;
        float *device_recovery;
        IzhikevichParameters* device_neuron_parameters;
#endif

};

#endif
