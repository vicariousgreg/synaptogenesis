#ifndef hebbian_rate_encoding_attributes_h
#define hebbian_rate_encoding_attributes_h

#include "state/impl/rate_encoding_attributes.h"

class HebbianRateEncodingAttributes : public RateEncodingAttributes {
    public:
        HebbianRateEncodingAttributes(Layer *layer);

        virtual KernelList<SYNAPSE_ARGS> get_updater(Connection *conn);

    ATTRIBUTE_MEMBERS
};

#endif
