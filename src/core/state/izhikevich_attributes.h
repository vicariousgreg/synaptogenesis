#ifndef izhikevich_attributes_h
#define izhikevich_attributes_h

#include "state/attributes.h"

/* Neuron parameters class.
 * Contains a,b,c,d parameters for Izhikevich model */
class IzhikevichParameters {
    public:
        IzhikevichParameters(float a, float b, float c, float d) :
                a(a), b(b), c(c), d(d) {}
        float a, b, c, d;
};

class IzhikevichAttributes : public Attributes {
    public:
        IzhikevichAttributes(Model* model);
        ~IzhikevichAttributes();

        void update(int start_index, int count);

        //////////////////////
        /// MODEL SPECIFIC ///
        //////////////////////
        // GETTERS
        /* If parallel, these will copy data from the device */
        float* get_voltage();
        float* get_recovery();

    private:
        friend class IzhikevichEngine;

        // Neuron Attributes
        float *voltage;
        float *recovery;

        // Neuron Spikes
        int* spikes;

        // Neuron parameters
        IzhikevichParameters* neuron_parameters;
};

#endif
