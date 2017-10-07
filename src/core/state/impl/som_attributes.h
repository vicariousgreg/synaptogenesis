#ifndef som_attributes_h
#define som_attributes_h

#include "state/attributes.h"

class SOMAttributes : public Attributes {
    public:
        SOMAttributes(LayerList &layers);

        virtual Kernel<SYNAPSE_ARGS> get_activator(Connection *conn);
        virtual Kernel<SYNAPSE_ARGS> get_updater(Connection *conn);

        Pointer<int> winner;
        Pointer<float> rbf_scale;
        Pointer<float> learning_rate;
        Pointer<float> neighbor_learning_rate;
        Pointer<int> neighborhood_size;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
