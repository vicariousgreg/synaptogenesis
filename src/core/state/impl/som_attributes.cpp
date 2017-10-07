#include <string>
#include <math.h>

#include "state/impl/som_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

#define DUMMY_VAL 1.0

REGISTER_ATTRIBUTES(SOMAttributes, "som", FLOAT)

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(SOMAttributes, som_attribute_kernel,
    SOMAttributes *som_att = (SOMAttributes*)att;
    float *f_outputs = (float*)outputs;
    int *winner = som_att->winner.get(layer_index);
    float rbf_scale = *som_att->rbf_scale.get(layer_index);

    float min = inputs[0];
    int min_node = 0;
    for (int i = 0 ; i < size ; ++i) {
        if (inputs[i] < min) {
            min = inputs[i];
            min_node = i;
        }
    }
    *winner = min_node;

    ,

    float next_value = f_outputs[nid];
    int index;
    for (index = 0 ; index < history_size-1 ; ++index) {
        float curr_value = next_value;
        next_value = f_outputs[size * (index + 1) + nid];
        f_outputs[size * index + nid] = next_value;
    }
    float input = inputs[nid];
    f_outputs[size * index + nid] = exp(-rbf_scale * (input * input));
)

/******************************************************************************/
/*************************** SOM ACTIVATOR KERNELS ****************************/
/******************************************************************************/

CALC_ALL(activate_som,
    ,

    float distance = 0.0;,

    float val = extractor(outputs[from_index], delay) - weights[weight_index];
    distance += val * val;,

    inputs[to_index] += distance;
);

CALC_ALL(update_som,
    SOMAttributes *som_att = (SOMAttributes*)synapse_data.attributes;
    float learning_rate =
        *som_att->learning_rate.get(synapse_data.connection_index);
    float neighbor_learning_rate =
        *som_att->neighbor_learning_rate.get(synapse_data.connection_index);
    int neighborhood_size =
        *som_att->neighborhood_size.get(synapse_data.connection_index);
    int winner = *som_att->winner.get(synapse_data.to_layer_index);

    int winner_row = winner / to_columns;
    int winner_col = winner % to_columns;,

    int row_dist = abs(winner_row - (to_index / to_columns));
    int col_dist = abs(winner_col - (to_index % to_columns));,

    if (to_index == winner)
        weights[weight_index] += learning_rate *
            (extractor(outputs[from_index], delay) - weights[weight_index]);
    else if (row_dist + col_dist <= neighborhood_size)
        weights[weight_index] += neighbor_learning_rate *
            (extractor(outputs[from_index], delay) - weights[weight_index]);,

);

Kernel<SYNAPSE_ARGS> SOMAttributes::get_activator(Connection *conn) {
    std::map<ConnectionType, Kernel<SYNAPSE_ARGS>> funcs;
   if (not conn->second_order)
        funcs[FULLY_CONNECTED] = get_activate_som_fully_connected();

    try {
        return funcs.at(conn->type);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Unimplemented connection type!");
    }
}

/******************************************************************************/
/**************************** SOM UPDATER KERNELS *****************************/
/******************************************************************************/

Kernel<SYNAPSE_ARGS> SOMAttributes::get_updater(Connection *conn) {
    std::map<ConnectionType, Kernel<SYNAPSE_ARGS>> funcs;
    if (not conn->second_order) {
        funcs[FULLY_CONNECTED] = get_update_som_fully_connected();
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

SOMAttributes::SOMAttributes(LayerList &layers)
        : Attributes(layers, FLOAT) {
    this->winner = Attributes::create_layer_variable<int>(0.0);
    Attributes::register_layer_variable("winner", &winner);

    this->rbf_scale = Attributes::create_layer_variable<float>(0.0);
    Attributes::register_layer_variable("rbf scale", &rbf_scale);

    this->learning_rate = Attributes::create_connection_variable<float>(0.0);
    Attributes::register_connection_variable("learning rate", &learning_rate);
    this->neighbor_learning_rate = Attributes::create_connection_variable<float>(0.0);
    Attributes::register_connection_variable("neighbor learning rate", &neighbor_learning_rate);
    this->neighborhood_size = Attributes::create_connection_variable<int>(0.0);
    Attributes::register_connection_variable("neighborhood size", &neighborhood_size);

    for (auto layer : layers) {
        rbf_scale[layer_indices[layer->id]] =
            std::stof(layer->get_parameter("rbf scale", "1"));

        for (auto conn : layer->get_input_connections()) {
            learning_rate[layer_indices[conn->id]] =
                std::stof(conn->get_parameter("learning rate", "0.01"));
            neighbor_learning_rate[layer_indices[conn->id]] =
                std::stof(conn->get_parameter("neighbor learning rate", "0.001"));
            neighborhood_size[layer_indices[conn->id]] =
                std::stoi(conn->get_parameter("neighborhood size", "2"));
        }
    }
}
