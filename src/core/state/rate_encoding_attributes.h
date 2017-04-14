#ifndef rate_encoding_attributes_h
#define rate_encoding_attributes_h

#include "state/attributes.h"

/* Neuron parameters class.
 * Contains parameters for Rate Encoding model */
class RateEncodingParameters {
    public:
        RateEncodingParameters(float x) : x(x) {}
        float x;
};

class RateEncodingAttributes : public Attributes {
    public:
        RateEncodingAttributes(LayerList &layers);

        // Neuron parameters
        Pointer<RateEncodingParameters> neuron_parameters;
};

#endif
