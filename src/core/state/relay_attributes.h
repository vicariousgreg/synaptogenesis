#ifndef relay_attributes_h
#define relay_attributes_h

#include "state/attributes.h"

class RelayAttributes : public Attributes {
    public:
        RelayAttributes(LayerList &layers);

    ATTRIBUTE_MEMBERS
};

#endif
