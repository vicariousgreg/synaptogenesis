#ifndef relay_attributes_h
#define relay_attributes_h

#include "state/attributes.h"

class RelayAttributes : public Attributes {
    public:
        RelayAttributes(LayerList &layers);
        static Attributes *build(LayerList &layers);

    private:
        static int neural_model_id;
};

#endif
