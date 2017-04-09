#ifndef relay_attributes_h
#define relay_attributes_h

#include "state/attributes.h"

class RelayAttributes : public Attributes {
    public:
        RelayAttributes(LayerList &layers);
        virtual ~RelayAttributes();

        virtual void schedule_transfer();
};

#endif
