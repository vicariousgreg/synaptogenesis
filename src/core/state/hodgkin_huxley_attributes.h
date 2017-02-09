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

GLOBAL void hh_attribute_kernel(const Attributes *att, int start_index, int count);

class HodgkinHuxleyAttributes : public SpikingAttributes {
    public:
        HodgkinHuxleyAttributes(Model* model);
        virtual ~HodgkinHuxleyAttributes();

        ATTRIBUTE_KERNEL get_attribute_kernel() {
            return hh_attribute_kernel;
        }

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
