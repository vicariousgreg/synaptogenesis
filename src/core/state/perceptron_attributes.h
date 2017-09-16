#ifndef perceptron_attributes_h
#define perceptron_attributes_h

#include "state/attributes.h"

class PerceptronAttributes : public Attributes {
    public:
        PerceptronAttributes(LayerList &layers);

        virtual Kernel<SYNAPSE_ARGS> get_activator(Connection *conn);
        virtual Kernel<SYNAPSE_ARGS> get_updater(Connection *conn);

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
