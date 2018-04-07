#ifndef oscillator_attributes_h
#define oscillator_attributes_h

#include "state/attributes.h"
#include "state/weight_matrix.h"

class NVMAttributes : public Attributes {
    public:
        NVMAttributes(Layer *layer);

        virtual bool check_compatibility(ClusterType cluster_type);

        virtual Kernel<SYNAPSE_ARGS> get_activator(Connection *conn);
        virtual Kernel<SYNAPSE_ARGS> get_updater(Connection *conn);

        virtual void process_weight_matrix(WeightMatrix* matrix);

        // Internal neural state
        Pointer<float> state;

        // Baseline tonic activity
        //float tonic;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class NVMWeightMatrix : public WeightMatrix {
    WEIGHT_MATRIX_MEMBERS(NVMWeightMatrix);
    virtual void register_variables();
};

#endif