#ifndef nvm_heaviside_attributes_h
#define nvm_heaviside_attributes_h

#include "state/impl/nvm_attributes.h"
#include "state/weight_matrix.h"

class NVMHeavisideAttributes : public NVMAttributes {
    public:
        NVMHeavisideAttributes(Layer *layer) : NVMAttributes(layer) { }

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
