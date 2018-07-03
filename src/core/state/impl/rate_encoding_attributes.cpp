#include <string>

#include "state/impl/rate_encoding_attributes.h"
#include "util/logger.h"

#include <math.h>

REGISTER_ATTRIBUTES(RateEncodingAttributes, "rate_encoding", FLOAT)

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(RateEncodingAttributes, re_attribute_kernel,
    float *f_outputs = (float*)outputs;

    ,

    float input = inputs[nid];
    SHIFT_FLOAT_OUTPUTS(f_outputs, (input > 0.0) ? tanh(input) : 0.0);
    // SHIFT_FLOAT_OUTPUTS(f_outputs, (input > 0.0) ? input : 0.0);
    // SHIFT_FLOAT_OUTPUTS(f_outputs, tanh(input));
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

KernelList<SYNAPSE_ARGS> RateEncodingAttributes::get_updater(Connection *conn) {
    LOG_ERROR("RateEncodingAttributes updater not implemented!");
}

RateEncodingAttributes::RateEncodingAttributes(Layer *layer)
        : Attributes(layer, FLOAT) { }
