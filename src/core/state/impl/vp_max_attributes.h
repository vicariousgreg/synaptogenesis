#ifndef vp_max_attributes_h
#define vp_max_attributes_h

#include "state/attributes.h"
#include "state/weight_matrix.h"

class VPMaxAttributes : public Attributes {
    public:
        VPMaxAttributes(Layer *layer);

        virtual bool check_compatibility(ClusterType cluster_type);

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);
        virtual KernelList<SYNAPSE_ARGS> get_updaters(Connection *conn);

        virtual void process_weight_matrix(WeightMatrix* matrix);

        // Internal neural state
        Pointer<float> state;

        // Baseline tonic activity
        //float tonic;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class VPMaxWeightMatrix : public WeightMatrix {
    WEIGHT_MATRIX_MEMBERS(VPMaxWeightMatrix);
    virtual void register_variables();
};

#endif
