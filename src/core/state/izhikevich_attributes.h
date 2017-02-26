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

GLOBAL void iz_attribute_kernel(const AttributeData attribute_data);

class IzhikevichAttributes : public SpikingAttributes {
    public:
        IzhikevichAttributes(LayerList &layers);
        virtual ~IzhikevichAttributes();

        virtual void transfer_to_device();

        // Neuron Attributes
        Pointer<float> recovery;

        // Neuron parameters
        Pointer<IzhikevichParameters> neuron_parameters;
};

#endif
