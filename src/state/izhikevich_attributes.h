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

#ifdef PARALLEL
        void update(int start_index, int count, cudaStream_t &stream);
#else
        void update(int start_index, int count);
#endif

        //////////////////////
        /// MODEL SPECIFIC ///
        //////////////////////
        // GETTERS
        /* If parallel, these will copy data from the device */
        float* get_voltage();
        float* get_recovery();

    private:
        friend class IzhikevichDriver;

        // Neuron Attributes
        float *voltage;
        float *recovery;

        // Neuron Spikes
        int* spikes;

        // Neuron parameters
        IzhikevichParameters* neuron_parameters;
};

#endif
