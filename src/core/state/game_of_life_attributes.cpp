#include <string>
#include <math.h>

#include "state/game_of_life_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(GameOfLifeAttributes, "game_of_life")

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(GameOfLifeAttributes, gol_attribute_kernel,

    ,

    int input = (int)inputs[nid];
    unsigned int *lives = (unsigned int*)outputs;
    bool prev_live = lives[nid] & 1;
    bool live = (input == 3) or (prev_live and (input == 2));

    /********************
     *** LIFE UPDATE ***
     ********************/
    // Reduce reads, chain values.
    unsigned int next_value = lives[nid];

    // Shift all the bits.
    // Check if next word is odd (1 for LSB).
    int index;
    for (index = 0 ; index < history_size-1 ; ++index) {
        unsigned int curr_value = next_value;
        next_value = lives[size * (index + 1) + nid];

        // Shift bits, carry over LSB from next value.
        lives[size*index + nid] = (curr_value >> 1) | (next_value << 31);
    }

    // Least significant value already loaded into next_value.
    // Index moved appropriately from loop.
    lives[size*index + nid] = (next_value >> 1) | (live << 31);
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

GameOfLifeAttributes::GameOfLifeAttributes(LayerList &layers)
        : Attributes(layers, BIT) { }
