#include "state/impl/ghost_attributes.h"
#include "util/logger.h"

REGISTER_ATTRIBUTES(GhostFloatAttributes, "ghost float", FLOAT)
REGISTER_ATTRIBUTES(GhostBitAttributes, "ghost bit", BIT)
REGISTER_ATTRIBUTES(GhostIntAttributes, "ghost int", INT)

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(GhostFloatAttributes, ghost_float_attribute_kernel,
    float *f_outputs = (float*)outputs;
    ,
    float input = inputs[nid];
    SHIFT_FLOAT_OUTPUTS(f_outputs, input);
)
BUILD_ATTRIBUTE_KERNEL(GhostBitAttributes, ghost_bit_attribute_kernel,
    unsigned int *b_outputs = (unsigned int*)outputs;
    ,
    unsigned int input = inputs[nid];
    SHIFT_BIT_OUTPUTS(b_outputs, input & 1);
)
BUILD_ATTRIBUTE_KERNEL(GhostIntAttributes, ghost_int_attribute_kernel,
    int *i_outputs = (int*)outputs;
    ,
    int input = inputs[nid];
    SHIFT_FLOAT_OUTPUTS(i_outputs, input);
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

GhostFloatAttributes::GhostFloatAttributes(Layer *layer) : Attributes(layer, FLOAT) { }
GhostBitAttributes::GhostBitAttributes(Layer *layer) : Attributes(layer, BIT) { }
GhostIntAttributes::GhostIntAttributes(Layer *layer) : Attributes(layer, INT) { }
