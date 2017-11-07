#include <string>

#include "state/impl/relay_attributes.h"
#include "util/error_manager.h"

REGISTER_ATTRIBUTES(RelayAttributes, "relay", FLOAT)

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(RelayAttributes, relay_attribute_kernel,
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
    f_outputs[size * index + nid] = input;
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

RelayAttributes::RelayAttributes(Layer *layer)
        : Attributes(layer, FLOAT) { }
