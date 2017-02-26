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
        RateEncodingAttributes(Structure* structure);
        virtual ~RateEncodingAttributes();

        virtual std::string get_stream_cluster_name() { return "feedforward"; }

        /* Trace learning rules */
        virtual SYNAPSE_KERNEL get_updater(ConnectionType type);

        virtual void transfer_to_device();

        // Neuron parameters
        Pointer<RateEncodingParameters> neuron_parameters;
};

#endif
