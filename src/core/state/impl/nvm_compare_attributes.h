#ifndef nvm_compare_attributes_h
#define nvm_compare_attributes_h

#include "state/impl/nvm_attributes.h"
#include "state/weight_matrix.h"

class NVMCompareAttributes : public NVMAttributes {
    public:
        NVMCompareAttributes(Layer *layer);

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);

        // "true" pattern for learning
        Pointer<float> true_state;

    ATTRIBUTE_MEMBERS
};

#endif
