#ifndef rate_encoding_attributes_h
#define rate_encoding_attributes_h

#include "state/state.h"

/* Neuron parameters class.
 * Contains parameters for Rate Encoding model */
class RateEncodingParameters {
    public:
        RateEncodingParameters(float x) : x(x) {}
        float x;
};

GLOBAL void re_attribute_kernel(const Attributes *att, int start_index, int count);

class RateEncodingAttributes : public Attributes {
    public:
        RateEncodingAttributes(Model* model);
        ~RateEncodingAttributes();

        ATTRIBUTE_KERNEL get_attribute_kernel() {
            return re_attribute_kernel;
        }

#ifdef PARALLEL
        virtual void send_to_device();
#endif

        // Neuron parameters
        RateEncodingParameters* neuron_parameters;
};

#endif
