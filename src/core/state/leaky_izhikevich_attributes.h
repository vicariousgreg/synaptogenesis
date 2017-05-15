#ifndef leaky_izhikevich_attributes_h
#define leaky_izhikevich_attributes_h

#include "state/izhikevich_attributes.h"

class LeakyIzhikevichAttributes : public IzhikevichAttributes {
    public:
        LeakyIzhikevichAttributes(LayerList &layers)
            : IzhikevichAttributes(layers) { }

        virtual Kernel<SYNAPSE_ARGS> get_updater(
            Connection *conn, DendriticNode *node);

    ATTRIBUTE_MEMBERS
};

#endif
