#ifndef sine_generator_attributes_h
#define sine_generator_attributes_h

#include "state/attributes.h"

class SineGeneratorAttributes : public Attributes {
    public:
        SineGeneratorAttributes(Layer *layer);

        int iteration;
        float frequency;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
