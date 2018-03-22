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
    // Cast the attributes pointer to the subclass type
    NVMLogisticAttributes *nvm_logistic_att = (NVMLogisticAttributes*)att;
    //float tonic = nvm_logistic_att->tonic;
    float* state = nvm_logistic_att->state.get();

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
    f_outputs[size * index + nid] = .5 * (tanh(st) + 1.); // logistic
)
