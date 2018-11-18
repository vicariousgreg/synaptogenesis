#ifndef ghost_attributes_h
#define ghost_attributes_h

#include "state/attributes.h"

class GhostFloatAttributes : public Attributes {
    public:
        GhostFloatAttributes(Layer *layer);

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class GhostBitAttributes : public Attributes {
    public:
        GhostBitAttributes(Layer *layer);

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class GhostIntAttributes : public Attributes {
    public:
        GhostIntAttributes(Layer *layer);

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
