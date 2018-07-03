#ifndef perceptron_attributes_h
#define perceptron_attributes_h

#include "state/attributes.h"

class PerceptronAttributes : public Attributes {
    public:
        PerceptronAttributes(Layer *layer);

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);
        virtual KernelList<SYNAPSE_ARGS> get_updaters(Connection *conn);

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
