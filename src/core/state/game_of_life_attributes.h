#ifndef game_of_life_attributes_h
#define game_of_life_attributes_h

#include "state/attributes.h"

class GameOfLifeAttributes : public Attributes {
    public:
        GameOfLifeAttributes(LayerList &layers);

        virtual int get_matrix_depth(Connection* conn) {
            /* Weight */
            return 1;
        }

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
