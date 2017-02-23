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
        HodgkinHuxleyAttributes(Structure* structure);
        virtual ~HodgkinHuxleyAttributes();

        ATTRIBUTE_KERNEL get_attribute_kernel() const {
            return hh_attribute_kernel;
        }

        virtual void transfer_to_device();

        // Neuron Attributes
        Pointer<float> *h;
        Pointer<float> *m;
        Pointer<float> *n;
        Pointer<float> *current_trace;

        // Neuron parameters
        Pointer<HodgkinHuxleyParameters> *neuron_parameters;
};

#endif
