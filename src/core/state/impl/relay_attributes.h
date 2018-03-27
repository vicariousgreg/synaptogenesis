#ifndef relay_attributes_h
#define relay_attributes_h

#include "state/attributes.h"

class RelayAttributes : public Attributes {
    public:
        RelayAttributes(Layer *layer);

    // Ramp (keep output positive)
    bool ramp;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
