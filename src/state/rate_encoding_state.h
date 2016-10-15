#ifndef rate_encoding_state_h
#define rate_encoding_state_h

#include "state/state.h"

/* Neuron parameters class.
 * Contains parameters for Rate Encoding model */
class RateEncodingParameters {
    public:
        RateEncodingParameters(float x) : x(x) {}
        float x;
};

class RateEncodingState : public State {
    public:
        void build(Model* model);

    private:
        friend class RateEncodingDriver;

        // Neuron parameters
        RateEncodingParameters* neuron_parameters;
};

#endif
