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
    float *context = ((NVMAttributes*)att)->context.get();
    float noise = ((NVMAttributes*)att)->noise_gate;
    float noise_offset = ((NVMAttributes*)att)->noise_offset;
    float gain = ((NVMAttributes*)att)->gain;
    ,
    float input = inputs[nid] + (noise ? rand - noise_offset : 0.);
    state[nid] = gain * ((input > 0) - (input <= 0));

    context[nid] = 1.;
    SHIFT_FLOAT_OUTPUTS(f_outputs, (float)(input > 0.));
)
