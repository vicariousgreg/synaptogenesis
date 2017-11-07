#ifndef som_attributes_h
#define som_attributes_h

#include "state/attributes.h"

class SOMAttributes : public Attributes {
    public:
        SOMAttributes(Layer *layer);

        virtual Kernel<SYNAPSE_ARGS> get_activator(Connection *conn);
        virtual Kernel<SYNAPSE_ARGS> get_updater(Connection *conn);

        virtual void process_weight_matrix(WeightMatrix* matrix);

        int winner;
        float rbf_scale;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class SOMWeightMatrix : public WeightMatrix {
    public:
        float learning_rate;
        float neighbor_learning_rate;
        int neighborhood_size;

    WEIGHT_MATRIX_MEMBERS(SOMWeightMatrix);
};

#endif
