#ifndef sample_attributes_h
#define sample_attributes_h

#include "state/attributes.h"
#include "state/weight_matrix.h"

class SampleAttributes : public Attributes {
    public:
        SampleAttributes(Layer *layer);

        virtual bool check_compatibility(ClusterType cluster_type);

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);
        virtual KernelList<SYNAPSE_ARGS> get_updaters(Connection *conn);

        virtual void process_weight_matrix(WeightMatrix* matrix);

        float layer_variable;
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
