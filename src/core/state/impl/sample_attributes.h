#ifndef sample_attributes_h
#define sample_attributes_h

#include "state/attributes.h"
#include "state/weight_matrix.h"

class SampleAttributes : public Attributes {
    public:
        SampleAttributes(LayerList &layers);

        virtual bool check_compatibility(ClusterType cluster_type);

        virtual Kernel<SYNAPSE_ARGS> get_activator(Connection *conn);
        virtual Kernel<SYNAPSE_ARGS> get_updater(Connection *conn);

        virtual void process_weight_matrix(WeightMatrix* matrix);

        Pointer<float> connection_variable;
        Pointer<float> layer_variable;
        Pointer<float> neuron_variable;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class SampleWeightMatrix : public WeightMatrix {
    public:
        Pointer<float> var1;
        Pointer<float> var2;
        float x;

    WEIGHT_MATRIX_MEMBERS(SampleWeightMatrix);
    virtual void register_variables();
};

#endif
