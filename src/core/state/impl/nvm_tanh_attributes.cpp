#include <string>
#include <math.h>

#include "state/impl/nvm_tanh_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(NVMTanhAttributes, "nvm_tanh", FLOAT)
USE_WEIGHT_MATRIX(NVMWeightMatrix, "nvm_tanh")

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(NVMTanhAttributes, nvm_tanh_kernel,
    // Cast the attributes pointer to the subclass type
    NVMTanhAttributes *nvm_tanh_att = (NVMTanhAttributes*)att;
    //float tonic = nvm_tanh_att->tonic;
    float* state = nvm_tanh_att->state.get();

    // input and output are automatically retrieved by the macro,
    //   but this casts the Output* to a float* for convenience
    float *f_outputs = (float*)outputs;

    ,

    float input = inputs[nid];
    float st = inputs[nid];
    state[nid] = st;
    SHIFT_FLOAT_OUTPUTS(f_outputs, tanh(st));
)
