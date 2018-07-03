#ifndef som_attributes_h
#define som_attributes_h

#include "state/attributes.h"

class SOMAttributes : public Attributes {
    public:
        SOMAttributes(Layer *layer);

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);
        virtual KernelList<SYNAPSE_ARGS> get_updaters(Connection *conn);

        virtual void process_weight_matrix(WeightMatrix* matrix);

        int winner;
        float rbf_scale;

        // Plasticity gate
        Pointer<float> plasticity;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class SOMWeightMatrix : public WeightMatrix {
    public:
        float learning_rate;
        float neighbor_learning_rate;
        int neighborhood_size;

        Pointer<float> output_cache;

    WEIGHT_MATRIX_MEMBERS(SOMWeightMatrix);
    virtual void register_variables();
};

#endif
