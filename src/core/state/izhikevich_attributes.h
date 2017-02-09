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

        virtual int get_matrix_depth() { return 3; }
        virtual void process_weight_matrix(WeightMatrix* matrix);

        virtual KERNEL get_activator(ConnectionType type) const {
            return get_activator_kernel_trace(type);
        }

        // Neuron Attributes
        float *voltage;
        float *recovery;

        // Neuron Current (copy of input)
        float* current;

        // Neuron Spikes (copy of output)
        unsigned int* spikes;

        // Neuron parameters
        IzhikevichParameters* neuron_parameters;
};

#endif
