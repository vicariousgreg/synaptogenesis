#include "state/impl/relay_attributes.h"
#include "util/logger.h"

REGISTER_ATTRIBUTES(RelayAttributes, "relay", FLOAT)

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(RelayAttributes, relay_attribute_kernel,
    float *f_outputs = (float*)outputs;
    bool ramp = ((RelayAttributes*)att)->ramp;

    ,

    float input = inputs[nid];
    SHIFT_FLOAT_OUTPUTS(f_outputs, ramp ? MAX(0.0, input) : input);
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

RelayAttributes::RelayAttributes(Layer *layer)
        : Attributes(layer, FLOAT) {
    this->ramp = layer->get_parameter("ramp", "false") == "true";
}
