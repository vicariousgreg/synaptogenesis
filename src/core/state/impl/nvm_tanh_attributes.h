#ifndef nvm_tanh_attributes_h
#define nvm_tanh_attributes_h

#include "state/impl/nvm_attributes.h"
#include "state/weight_matrix.h"

class NVMTanhAttributes : public NVMAttributes {
    public:
        NVMTanhAttributes(Layer *layer) : NVMAttributes(layer) { }

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
