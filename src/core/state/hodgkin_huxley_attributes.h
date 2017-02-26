#ifndef hodgkin_huxley_attributes_h
#define hodgkin_huxley_attributes_h

#include "state/spiking_attributes.h"

/* Neuron parameters class.
 * Contains h,n,m parameters for Hodgkin-Huxley model */
class HodgkinHuxleyParameters {
    public:
        HodgkinHuxleyParameters(float iapp) : iapp(iapp) {}
        float iapp;
};

GLOBAL void hh_attribute_kernel(const AttributeData attribute_data);

class HodgkinHuxleyAttributes : public SpikingAttributes {
    public:
        HodgkinHuxleyAttributes(LayerList &layers);
        virtual ~HodgkinHuxleyAttributes();

        virtual void transfer_to_device();

        // Neuron Attributes
        Pointer<float> h, m, n, current_trace;

        // Neuron parameters
        Pointer<HodgkinHuxleyParameters> neuron_parameters;
};

#endif
