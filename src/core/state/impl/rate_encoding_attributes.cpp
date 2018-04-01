#include <string>

#include "state/impl/rate_encoding_attributes.h"
#include "util/logger.h"

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

#include <math.h>

BUILD_ATTRIBUTE_KERNEL(RateEncodingAttributes, re_attribute_kernel,
    float *f_outputs = (float*)outputs;

    ,

    float input = inputs[nid];
    SHIFT_FLOAT_OUTPUTS(f_outputs, (input > 0.0) ? tanh(0.1*input) : 0.0);
    // SHIFT_FLOAT_OUTPUTS(f_outputs, (input > 0.0) ? input : 0.0);
    // SHIFT_FLOAT_OUTPUTS(f_outputs, tanh(input));
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

RateEncodingAttributes::RateEncodingAttributes(Layer *layer)
        : Attributes(layer, FLOAT) { }
