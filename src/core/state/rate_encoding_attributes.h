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

class RateEncodingAttributes : public Attributes {
    public:
        RateEncodingAttributes(Model* model);
        ~RateEncodingAttributes();

        int get_matrix_depth() { return 3; }

        // Neuron parameters
        RateEncodingParameters* neuron_parameters;
};

#endif
