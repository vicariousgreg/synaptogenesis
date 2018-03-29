#include <string>
#include <math.h>

#include "state/impl/game_of_life_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(GameOfLifeAttributes, "game_of_life", BIT)

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(GameOfLifeAttributes, gol_attribute_kernel,
    int survival_min = att->survival_min;
    int survival_max = att->survival_max;
    int birth_min = att->birth_min;
    int birth_max = att->birth_max;

    ,

    int input = (int)inputs[nid];
    unsigned int *lives = (unsigned int*)outputs;
    bool prev_live = lives[nid] & 1;
    bool live = (prev_live)
        ? (input >= survival_min and input <= survival_max)
        : (input >= birth_min and input <= birth_max);

    /********************
     *** LIFE UPDATE ***
     ********************/
    SHIFT_BIT_OUTPUTS(lives, live);
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

GameOfLifeAttributes::GameOfLifeAttributes(Layer *layer)
        : Attributes(layer, BIT) {
    this->survival_min = std::stoi(layer->get_parameter("survival_min", "2"));
    this->survival_max = std::stoi(layer->get_parameter("survival_max", "3"));
    this->birth_min = std::stoi(layer->get_parameter("birth_min", "3"));
    this->birth_max = std::stoi(layer->get_parameter("birth_max", "3"));
}
