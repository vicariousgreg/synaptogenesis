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
        virtual ~RateEncodingAttributes();

        /* Checks whether these attributes are compatible
         *   with the given cluster_type */
        virtual bool check_compatibility(ClusterType cluster_type) {
            return cluster_type == FEEDFORWARD;
        }

        virtual void schedule_transfer();

        // Neuron parameters
        Pointer<RateEncodingParameters> neuron_parameters;
};

#endif
