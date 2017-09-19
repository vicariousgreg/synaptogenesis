#ifndef backprop_rate_encoding_attributes_h
#define backprop_rate_encoding_attributes_h

#include "state/impl/rate_encoding_attributes.h"

class BackpropRateEncodingAttributes : public RateEncodingAttributes {
    public:
        BackpropRateEncodingAttributes(LayerList &layers);

        virtual bool check_compatibility(ClusterType cluster_type) {
            return cluster_type == FEEDFORWARD;
        }

        virtual Kernel<SYNAPSE_ARGS> get_updater(
            ConnectionType type, DendriticNode *node);

        Pointer<float> error_deltas;

    ATTRIBUTE_MEMBERS
};

#endif