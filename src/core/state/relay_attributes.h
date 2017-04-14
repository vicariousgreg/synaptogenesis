#ifndef relay_attributes_h
#define relay_attributes_h

#include "state/attributes.h"

class RelayAttributes : public Attributes {
    public:
        RelayAttributes(LayerList &layers);

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
