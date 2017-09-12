#ifndef debug_attributes_h
#define debug_attributes_h

#include "state/attributes.h"

class DebugAttributes : public Attributes {
    public:
        DebugAttributes(LayerList &layers);

        virtual Kernel<SYNAPSE_ARGS> get_activator(Connection *conn);
        virtual Kernel<SYNAPSE_ARGS> get_updater(Connection *conn);

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
