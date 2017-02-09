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
        HodgkinHuxleyAttributes(Model* model);
        ~HodgkinHuxleyAttributes();

#ifdef PARALLEL
        virtual void send_to_device();
#endif

        // Neuron Attributes
        float *h, *m, *n;
        float* current_trace;

        // Neuron parameters
        HodgkinHuxleyParameters* neuron_parameters;
};

#endif
