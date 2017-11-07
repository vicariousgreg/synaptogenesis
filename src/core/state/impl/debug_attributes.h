#ifndef debug_attributes_h
#define debug_attributes_h

#include "state/attributes.h"

class DebugAttributes : public Attributes {
    public:
        DebugAttributes(LayerList &layers);

        virtual Kernel<SYNAPSE_ARGS> get_activator(Connection *conn);
        virtual Kernel<SYNAPSE_ARGS> get_updater(Connection *conn);

        virtual void process_weight_matrix(WeightMatrix* matrix);

        Pointer<float> connection_variable;
        Pointer<float> layer_variable;
        Pointer<float> neuron_variable;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class DebugWeightMatrix : public WeightMatrix {
    public:
        float x;

    WEIGHT_MATRIX_MEMBERS(DebugWeightMatrix);
};

#endif
