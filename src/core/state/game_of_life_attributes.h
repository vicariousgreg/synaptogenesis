#ifndef game_of_life_attributes_h
#define game_of_life_attributes_h

#include "state/attributes.h"

class GameOfLifeAttributes : public Attributes {
    public:
        GameOfLifeAttributes(LayerList &layers);

    Pointer<int> survival_mins;
    Pointer<int> survival_maxs;
    Pointer<int> birth_mins;
    Pointer<int> birth_maxs;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
