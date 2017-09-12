#include <string>
#include <math.h>

#include "state/debug_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(DebugAttributes, "debug")

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(DebugAttributes, debug_attribute_kernel,
    DebugAttributes *debug_att = (DebugAttributes*)att;

    ,

    assert(nid < size);
)

/******************************************************************************/
/************************* DEBUG ACTIVATOR KERNELS ****************************/
/******************************************************************************/

CALC_ALL(activate_debug,
    ,

    assert(to_index < to_size);
    Output to_out = destination_outputs[to_index];,

    assert(from_index < from_size);
    Output from_out = outputs[from_index];
    assert(weight_index < num_weights);
    float weight = weights[weight_index];,
);
CALC_ONE_TO_ONE(activate_debug_convolutional_second_order,
    ,

    /* Don't check to_index for convolutional second order kernel */
    ,

    assert(from_index < from_size);
    Output from_out = outputs[from_index];
    assert(weight_index < num_weights);
    float weight = weights[weight_index];,

);

CALC_ALL(update_debug,
    ,

    assert(to_index < to_size);
    Output to_out = destination_outputs[to_index];,

    assert(from_index < from_size);
    Output from_out = outputs[from_index];
    assert(weight_index < num_weights);
    float weight = weights[weight_index];,
);
CALC_CONVOLUTIONAL_BY_WEIGHT(update_debug_convolutional,
    ,

    float weight = weights[weight_index];
    assert(weight_index < num_weights);,

    assert(from_index < from_size);
    Output from_out = outputs[from_index];
    assert(to_index < to_size);
    Output to_out = destination_outputs[to_index];,
);

Kernel<SYNAPSE_ARGS> DebugAttributes::get_activator(Connection *conn) {
    std::map<ConnectionType, Kernel<SYNAPSE_ARGS> > funcs;
    if (conn->second_order) {
        funcs[FULLY_CONNECTED]      = get_update_debug_fully_connected();
        funcs[SUBSET]               = get_update_debug_subset();
        funcs[ONE_TO_ONE]           = get_update_debug_one_to_one();
        funcs[CONVERGENT]           = get_update_debug_convergent();
        funcs[CONVOLUTIONAL]        = get_update_debug_convolutional();
        funcs[DIVERGENT]            = get_update_debug_divergent();
    } else {
        funcs[FULLY_CONNECTED]      = get_activate_debug_fully_connected();
        funcs[SUBSET]               = get_activate_debug_subset();
        funcs[ONE_TO_ONE]           = get_activate_debug_one_to_one();
        funcs[CONVERGENT]           = get_activate_debug_convergent();
        funcs[CONVOLUTIONAL]        = get_activate_debug_convergent();
        funcs[DIVERGENT]            = get_activate_debug_divergent();
    }

    try {
        return funcs.at(conn->type);
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** DEBUG UPDATER KERNELS *****************************/
/******************************************************************************/

Kernel<SYNAPSE_ARGS> DebugAttributes::get_updater(Connection *conn) {
    std::map<ConnectionType, Kernel<SYNAPSE_ARGS> > funcs;
    if (not conn->second_order) {
        funcs[FULLY_CONNECTED]      = get_update_debug_fully_connected();
        funcs[SUBSET]               = get_update_debug_subset();
        funcs[ONE_TO_ONE]           = get_update_debug_one_to_one();
        funcs[CONVERGENT]           = get_update_debug_convergent();
        funcs[CONVOLUTIONAL]        = get_update_debug_convolutional();
        funcs[DIVERGENT]            = get_update_debug_divergent();
    }

    try {
        return funcs.at(conn->type);
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

DebugAttributes::DebugAttributes(LayerList &layers)
        : Attributes(layers, FLOAT) { }
