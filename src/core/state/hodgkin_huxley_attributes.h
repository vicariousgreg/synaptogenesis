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

class HodgkinHuxleyAttributes : public SpikingAttributes {
    public:
        HodgkinHuxleyAttributes(LayerList &layers);

        // Neuron Attributes
        Pointer<float> h, m, n, current_trace;

        // Neuron parameters
        Pointer<HodgkinHuxleyParameters> neuron_parameters;

    ATTRIBUTE_MEMBERS
};

#endif
