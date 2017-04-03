#ifndef backprop_rate_encoding_attributes_h
#define backprop_rate_encoding_attributes_h

#include "state/rate_encoding_attributes.h"

class BackpropRateEncodingAttributes : public RateEncodingAttributes {
    public:
        BackpropRateEncodingAttributes(LayerList &layers);
        virtual ~BackpropRateEncodingAttributes();

        virtual Kernel<SYNAPSE_ARGS> get_updater(
            ConnectionType type, bool second_order);

        virtual void schedule_transfer();

        Pointer<float> error_deltas;
};

#endif
