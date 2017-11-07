#include <string>
#include <math.h>

#include "state/impl/debug_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

#define DUMMY_VAL 1.0

REGISTER_ATTRIBUTES(DebugAttributes, "debug", FLOAT)
REGISTER_WEIGHT_MATRIX(DebugWeightMatrix, "debug")

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(DebugAttributes, debug_attribute_kernel,
    DebugAttributes *debug_att = (DebugAttributes*)att;

    assert(layer_index < debug_att->layer_variable.get_size());
    float layer_var = *debug_att->layer_variable.get(layer_index);
    assert(layer_var == DUMMY_VAL);

    assert(other_start_index < debug_att->neuron_variable.get_size());
    assert((other_start_index + size) <= debug_att->neuron_variable.get_size());
    float *neuron_var = debug_att->neuron_variable.get(other_start_index);

    ,

    assert(nid < size);
    assert(neuron_var[nid] == DUMMY_VAL);
)

/******************************************************************************/
/************************* DEBUG ACTIVATOR KERNELS ****************************/
/******************************************************************************/

#define CHECK_ATT \
    DebugAttributes *debug_att = (DebugAttributes*)synapse_data.attributes; \
    DebugWeightMatrix *debug_mat = (DebugWeightMatrix*)synapse_data.matrix; \
    assert(debug_mat->x == DUMMY_VAL); \
    assert(from_rows * from_columns == from_size); \
    assert(to_rows * to_columns == to_size);

#define CHECK_TO \
    assert(to_index < to_size); \
    float to_in = inputs[to_index]; \
    Output to_out = destination_outputs[to_index];

#define CHECK_FROM \
    assert(from_index < from_size); \
    Output from_out = outputs[from_index];

#define CHECK_WEIGHT \
    assert(weight_index < num_weights); \
    float weight = weights[weight_index];

CALC_ALL(activate_debug,
    CHECK_ATT,

    CHECK_TO,

    CHECK_FROM
    CHECK_WEIGHT,
);
CALC_ONE_TO_ONE(activate_debug_convolutional_second_order,
    CHECK_ATT
    assert(synapse_data.connection.convolutional),

    /* Don't check to_index for convolutional second order kernel */
    ,

    CHECK_FROM
    CHECK_WEIGHT,
);


CALC_ALL(update_debug,
    CHECK_ATT,

    CHECK_TO,

    CHECK_FROM
    CHECK_WEIGHT,
);
CALC_CONVERGENT_CONVOLUTIONAL_BY_WEIGHT(update_debug_convergent_convolutional,
    CHECK_ATT
    assert(synapse_data.connection.convolutional);,

    CHECK_WEIGHT,

    CHECK_FROM
    CHECK_TO,
);
CALC_DIVERGENT_CONVOLUTIONAL_BY_WEIGHT(update_debug_divergent_convolutional,
    CHECK_ATT
    assert(synapse_data.connection.convolutional);,

    CHECK_WEIGHT,

    CHECK_FROM
    CHECK_TO,
);


Kernel<SYNAPSE_ARGS> DebugAttributes::get_activator(Connection *conn) {
    std::map<ConnectionType, Kernel<SYNAPSE_ARGS>> funcs;
    if (conn->second_order) {
        funcs[FULLY_CONNECTED]      = get_update_debug_fully_connected();
        funcs[SUBSET]               = get_update_debug_subset();
        funcs[ONE_TO_ONE]           = get_update_debug_one_to_one();
        funcs[CONVERGENT]           = (conn->convolutional)
                                          ? get_activate_debug_convolutional_second_order()
                                          : get_update_debug_convergent();
        funcs[DIVERGENT]            = (conn->convolutional)
                                          ? get_activate_debug_convolutional_second_order()
                                          : get_update_debug_divergent();
    } else {
        funcs[FULLY_CONNECTED]      = get_activate_debug_fully_connected();
        funcs[SUBSET]               = get_activate_debug_subset();
        funcs[ONE_TO_ONE]           = get_activate_debug_one_to_one();
        funcs[CONVERGENT]           = get_activate_debug_convergent();
        funcs[DIVERGENT]            = get_activate_debug_divergent();
    }

    try {
        return funcs.at(conn->type);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** DEBUG UPDATER KERNELS *****************************/
/******************************************************************************/
Kernel<SYNAPSE_ARGS> DebugAttributes::get_updater(Connection *conn) {
    std::map<ConnectionType, Kernel<SYNAPSE_ARGS>> funcs;
    if (not conn->second_order) {
        funcs[FULLY_CONNECTED] = get_update_debug_fully_connected();
        funcs[SUBSET]          = get_update_debug_subset();
        funcs[ONE_TO_ONE]      = get_update_debug_one_to_one();
        funcs[CONVERGENT]      = (conn->convolutional)
                                     ? get_update_debug_convergent_convolutional()
                                     : get_update_debug_convergent();
        funcs[DIVERGENT]       = (conn->convolutional)
                                     ? get_update_debug_divergent_convolutional()
                                     : get_update_debug_divergent();
    }

    try {
        return funcs.at(conn->type);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

DebugAttributes::DebugAttributes(LayerList &layers)
        : Attributes(layers, FLOAT) {
    this->layer_variable = Attributes::create_layer_variable<float>(DUMMY_VAL);
    Attributes::register_layer_variable("layer_var", &layer_variable);

    this->neuron_variable = Attributes::create_neuron_variable<float>(DUMMY_VAL);
    Attributes::register_neuron_variable("neuron_var", &neuron_variable);
}

void DebugAttributes::process_weight_matrix(WeightMatrix* matrix) {
    ((DebugWeightMatrix*)matrix)->x = DUMMY_VAL;
}
