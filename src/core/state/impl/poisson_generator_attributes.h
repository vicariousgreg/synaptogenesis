#ifndef poisson_generator_attributes_h
#define poisson_generator_attributes_h

#include "state/attributes.h"

class PoissonGeneratorAttributes : public Attributes {
    public:
        PoissonGeneratorAttributes(Layer *layer);

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
