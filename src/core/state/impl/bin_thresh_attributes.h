#ifndef bin_thresh_attributes_h
#define bin_thresh_attributes_h

#include "state/attributes.h"

class BinaryThresholdAttributes : public Attributes {
    public:
        BinaryThresholdAttributes(Layer *layer) : Attributes(layer, FLOAT) { }

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
