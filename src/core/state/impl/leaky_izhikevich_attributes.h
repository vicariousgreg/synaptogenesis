#ifndef leaky_izhikevich_attributes_h
#define leaky_izhikevich_attributes_h

#include "state/impl/izhikevich_attributes.h"

class LeakyIzhikevichAttributes : public IzhikevichAttributes {
    public:
        LeakyIzhikevichAttributes(Layer *layer)
            : IzhikevichAttributes(layer) { }

        virtual Kernel<SYNAPSE_ARGS> get_updater(Connection *conn);

    ATTRIBUTE_MEMBERS
};

#endif
