#include <string>
#include <math.h>

#include "state/impl/nvm_logistic_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(NVMLogisticAttributes, "nvm_logistic", FLOAT)
USE_WEIGHT_MATRIX(NVMWeightMatrix, "nvm_logistic")

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(NVMLogisticAttributes, nvm_logistic_kernel,
    //float tonic = att->tonic;
    float* state = att->state.get();

    // input and output are automatically retrieved by the macro,
    //   but this casts the Output* to a float* for convenience
    float *f_outputs = (float*)outputs;

    ,

    float input = inputs[nid];
    float st = inputs[nid];
    state[nid] = st;
    SHIFT_FLOAT_OUTPUTS(f_outputs, .5 * (tanh(st) + 1.)); // logistic
)
