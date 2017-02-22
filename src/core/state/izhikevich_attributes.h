#ifndef izhikevich_attributes_h
#define izhikevich_attributes_h

#include "state/spiking_attributes.h"

/* Neuron parameters class.
 * Contains a,b,c,d parameters for Izhikevich model */
class IzhikevichParameters {
    public:
        IzhikevichParameters(float a, float b, float c, float d) :
                a(a), b(b), c(c), d(d) {}
        float a, b, c, d;
};

GLOBAL void iz_attribute_kernel(const Attributes *att, int start_index, int count);

class IzhikevichAttributes : public SpikingAttributes {
    public:
        IzhikevichAttributes(Structure* structure);
        virtual ~IzhikevichAttributes();

        ATTRIBUTE_KERNEL get_attribute_kernel() const {
            return iz_attribute_kernel;
        }

#ifdef PARALLEL
        virtual void send_to_device();
#endif

        // Neuron Attributes
        float *recovery;

        // Neuron parameters
        IzhikevichParameters* neuron_parameters;
};

#endif
