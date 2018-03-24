#include "state/impl/poisson_generator_attributes.h"
#include "util/error_manager.h"

REGISTER_ATTRIBUTES(PoissonGeneratorAttributes, "poisson generator", BIT)

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_RAND_ATTRIBUTE_KERNEL(PoissonGeneratorAttributes, poisson_generator_kernel,
    unsigned int *spikes = (unsigned int*)outputs;

    ,

    /********************
     *** SPIKE UPDATE ***
     ********************/
    // Determine if spike occurred
    unsigned int spike = rand < inputs[nid];

    // Reduce reads, chain values.
    unsigned int next_value = spikes[nid];

    // Shift all the bits.
    // Check if next word is odd (1 for LSB).
    int index;
    for (index = 0 ; index < history_size-1 ; ++index) {
        unsigned int curr_value = next_value;
        next_value = spikes[size * (index + 1) + nid];

        // Shift bits, carry over LSB from next value.
        spikes[size*index + nid] = (curr_value >> 1) | (next_value << 31);
    }

    // Least significant value already loaded into next_value.
    // Index moved appropriately from loop.
    spikes[size*index + nid] = (next_value >> 1) | (spike << 31);
    bool prev_spike = next_value >> 31;
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

PoissonGeneratorAttributes::PoissonGeneratorAttributes(Layer *layer)
        : Attributes(layer, BIT) { }
