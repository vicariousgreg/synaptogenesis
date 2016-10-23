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
        RateEncodingAttributes(Model* model);
        ~RateEncodingAttributes();

#ifdef PARALLEL
        void update(int start_index, int count, cudaStream_t &stream);
#else
        void update(int start_index, int count);
#endif

    private:
        friend class RateEncodingDriver;

        // Neuron parameters
        RateEncodingParameters* neuron_parameters;
};

#endif
