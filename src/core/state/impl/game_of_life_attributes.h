#ifndef game_of_life_attributes_h
#define game_of_life_attributes_h

#include "state/attributes.h"

class GameOfLifeAttributes : public Attributes {
    public:
        GameOfLifeAttributes(Layer *layer);

    int survival_min;
    int survival_max;
    int birth_min;
    int birth_max;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
