#include <string>

#include "state/impl/rate_encoding_attributes.h"
#include "util/error_manager.h"

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

#include <math.h>

BUILD_ATTRIBUTE_KERNEL(RateEncodingAttributes, re_attribute_kernel,
    float *f_outputs = (float*)outputs;

    ,

    float next_value = f_outputs[nid];
    int index;
    for (index = 0 ; index < history_size-1 ; ++index) {
        float curr_value = next_value;
        next_value = f_outputs[size * (index + 1) + nid];
        f_outputs[size * index + nid] = next_value;
    }
    float input = inputs[nid];
    f_outputs[size * index + nid] =
        (input > 0.0) ? tanh(0.1*input) : 0.0;
        //(input > 0.0) ? input : 0.0;
        //tanh(input);
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

RateEncodingAttributes::RateEncodingAttributes(Layer *layer)
        : Attributes(layer, FLOAT) { }
