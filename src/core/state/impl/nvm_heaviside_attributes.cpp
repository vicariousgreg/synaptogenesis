#include "state/impl/nvm_heaviside_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"

REGISTER_ATTRIBUTES(NVMHeavisideAttributes, "nvm_heaviside", FLOAT)
USE_WEIGHT_MATRIX(NVMWeightMatrix, "nvm_heaviside")

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_RAND_ATTRIBUTE_KERNEL(NVMHeavisideAttributes, nvm_heaviside_kernel,
    float* state = att->state.get();
    float *f_outputs = (float*)outputs;
    float noise_scale = ((NVMAttributes*)att)->noise_scale;
    float noise_offset = ((NVMAttributes*)att)->noise_offset;
    ,
    float input = inputs[nid] + (rand * noise_scale - noise_offset);
    state[nid] = input;
    SHIFT_FLOAT_OUTPUTS(f_outputs, (float)(input > 0.));
)
