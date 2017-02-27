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

GLOBAL void re_attribute_kernel(const AttributeData attribute_data);

class RateEncodingAttributes : public Attributes {
    public:
        RateEncodingAttributes(LayerList &layers);
        virtual ~RateEncodingAttributes();

        /* Checks whether these attributes are compatible
         *   with the given cluster_type */
        virtual bool check_compatibility(ClusterType cluster_type) {
            return cluster_type == FEEDFORWARD;
        }

        /* Trace learning rules */
        virtual SYNAPSE_KERNEL get_updater(ConnectionType type);

        virtual void transfer_to_device();

        // Neuron parameters
        Pointer<RateEncodingParameters> neuron_parameters;
};

#endif
