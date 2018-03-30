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
    float layer_var = att->layer_variable;
    assert(layer_var == DUMMY_VAL);

    float *neuron_var = att->neuron_variable.get();

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
CALC_ONE_TO_ONE(activate_debug_second_order_convolutional,
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
    if (conn->convolutional and conn->second_order) {
        return get_activate_debug_second_order_convolutional();
    }

    try {
        return activate_debug_map.at(conn->get_type());
    } catch(std::out_of_range) { }

    // Log an error if the connection type is unimplemented
    LOG_ERROR("Unimplemented connection type!");
}

/******************************************************************************/
/************************** DEBUG UPDATER KERNELS *****************************/
/******************************************************************************/
Kernel<SYNAPSE_ARGS> DebugAttributes::get_updater(Connection *conn) {
    if (conn->second_order)
        LOG_ERROR("Unimplemented connection type!");

    if (conn->convolutional) {
        if (conn->get_type() == CONVERGENT)
            return get_update_debug_convergent_convolutional();
        else if (conn->get_type() == DIVERGENT)
            return get_update_debug_divergent_convolutional();
    }

    try {
        return update_debug_map.at(conn->get_type());
    } catch(std::out_of_range) { }

    // Log an error if the connection type is unimplemented
    LOG_ERROR("Unimplemented connection type!");
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

DebugAttributes::DebugAttributes(Layer *layer)
        : Attributes(layer, FLOAT) {
    this->layer_variable = DUMMY_VAL;

    this->neuron_variable = Attributes::create_neuron_variable<float>(DUMMY_VAL);
    Attributes::register_neuron_variable("neuron_var", &neuron_variable);
}

void DebugAttributes::process_weight_matrix(WeightMatrix* matrix) {
    ((DebugWeightMatrix*)matrix)->x = DUMMY_VAL;
}
