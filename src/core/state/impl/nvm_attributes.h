#ifndef oscillator_attributes_h
#define oscillator_attributes_h

#include "state/attributes.h"
#include "state/weight_matrix.h"

class NVMAttributes : public Attributes {
    public:
        NVMAttributes(Layer *layer);

        virtual bool check_compatibility(ClusterType cluster_type)
            { return cluster_type == PARALLEL; }

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);

        virtual void process_weight_matrix(WeightMatrix* matrix);

        // Internal neural state
        Pointer<float> state;

        // Gate registers
        bool activity_gate, learning_gate;

        // Normalization factor for learning
        float norm;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class NVMWeightMatrix : public WeightMatrix {
    public:
        float norm;

    WEIGHT_MATRIX_MEMBERS(NVMWeightMatrix);
};

#endif
