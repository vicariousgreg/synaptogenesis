#include <string>
#include <math.h>

#include "state/impl/nvm_heaviside_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(NVMHeavisideAttributes, "nvm_heaviside", FLOAT)
USE_WEIGHT_MATRIX(NVMWeightMatrix, "nvm_heaviside")

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(NVMHeavisideAttributes, nvm_heaviside_kernel,
    // Cast the attributes pointer to the subclass type
    NVMHeavisideAttributes *nvm_heaviside_att = (NVMHeavisideAttributes*)att;
    //float tonic = nvm_heaviside_att->tonic;
    float* state = nvm_heaviside_att->state.get();

    // input and output are automatically retrieved by the macro,
    //   but this casts the Output* to a float* for convenience
    float *f_outputs = (float*)outputs;

    ,

    // If you want to respect delays, this is the algorithm
    // If not, delayed output connections will get garbage data
    float next_value = f_outputs[nid];
    int index;
    for (index = 0 ; index < history_size-1 ; ++index) {
        float curr_value = next_value;
        next_value = f_outputs[size * (index + 1) + nid];
        f_outputs[size * index + nid] = next_value;
    }
    float input = inputs[nid];

    // This is the appropriate index to use for the most recent output
    next_value = f_outputs[size * index + nid];
    float st = inputs[nid];
    state[nid] = st;
    f_outputs[size * index + nid] = (float)(st > 0.);
    // f_outputs[size * index + nid] = .5 * (tanh(st) + 1.); // logistic
)
