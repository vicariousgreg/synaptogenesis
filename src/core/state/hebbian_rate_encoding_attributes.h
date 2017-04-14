#ifndef hebbian_rate_encoding_attributes_h
#define hebbian_rate_encoding_attributes_h

#include "state/rate_encoding_attributes.h"

class HebbianRateEncodingAttributes : public RateEncodingAttributes {
    public:
        HebbianRateEncodingAttributes(LayerList &layers);

        virtual Kernel<SYNAPSE_ARGS> get_updater(
            ConnectionType type, bool second_order);
};

#endif
