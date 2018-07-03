#ifndef debug_attributes_h
#define debug_attributes_h

#include "state/attributes.h"

class DebugAttributes : public Attributes {
    public:
        DebugAttributes(Layer *layer);

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);
        virtual KernelList<SYNAPSE_ARGS> get_updaters(Connection *conn);

        virtual void process_weight_matrix(WeightMatrix* matrix);

        Pointer<float> connection_variable;
        Pointer<float> neuron_variable;

        float layer_variable;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class DebugWeightMatrix : public WeightMatrix {
    public:
        float x;

    WEIGHT_MATRIX_MEMBERS(DebugWeightMatrix);
};

#endif
