#ifndef nvm_compare_attributes_h
#define nvm_compare_attributes_h

#include "state/impl/nvm_heaviside_attributes.h"
#include "state/weight_matrix.h"

class NVMCompareAttributes : public NVMHeavisideAttributes {
    public:
        NVMCompareAttributes(Layer *layer);

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);

        // "true" pattern for learning
        Pointer<float> true_state;

        // Comparison tolerance
        // Expressed as percentage similarity
        float tolerance;

    ATTRIBUTE_MEMBERS
};

#endif
