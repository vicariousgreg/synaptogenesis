#ifndef rate_encoding_attributes_h
#define rate_encoding_attributes_h

#include "state/attributes.h"

class RateEncodingAttributes : public Attributes {
    public:
        RateEncodingAttributes(LayerList &layers);

    GET_KERNEL_DEF
};

#endif
