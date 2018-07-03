#ifndef rate_encoding_attributes_h
#define rate_encoding_attributes_h

#include "state/attributes.h"

class RateEncodingAttributes : public Attributes {
    public:
        RateEncodingAttributes(Layer *layer);

        virtual KernelList<SYNAPSE_ARGS> get_updater(Connection *conn);

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
