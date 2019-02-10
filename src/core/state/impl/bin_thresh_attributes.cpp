#include "state/impl/bin_thresh_attributes.h"
#include "util/logger.h"

REGISTER_ATTRIBUTES(BinaryThresholdAttributes, "binary threshold", FLOAT)

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(BinaryThresholdAttributes, bin_thresh_attribute_kernel,
    float *f_outputs = (float*)outputs;

    ,

    float input = inputs[nid];
    SHIFT_FLOAT_OUTPUTS(f_outputs, input > 0);
)
