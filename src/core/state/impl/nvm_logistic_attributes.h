#ifndef nvm_logistic_attributes_h
#define nvm_logistic_attributes_h

#include "state/impl/nvm_attributes.h"
#include "state/weight_matrix.h"

class NVMLogisticAttributes : public NVMAttributes {
    public:
        NVMLogisticAttributes(Layer *layer) : NVMAttributes(layer) { }

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
