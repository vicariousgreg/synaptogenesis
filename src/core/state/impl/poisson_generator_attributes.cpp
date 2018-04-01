#include "state/impl/poisson_generator_attributes.h"
#include "util/logger.h"

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

    SHIFT_BIT_OUTPUTS(spikes, spike);
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

PoissonGeneratorAttributes::PoissonGeneratorAttributes(Layer *layer)
        : Attributes(layer, BIT) { }
