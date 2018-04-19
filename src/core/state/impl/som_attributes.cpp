#include <string>
#include <cmath>

#include "state/impl/som_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"

#define DUMMY_VAL 1.0

REGISTER_ATTRIBUTES(SOMAttributes, "som", FLOAT)
REGISTER_WEIGHT_MATRIX(SOMWeightMatrix, "som")

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(SOMAttributes, som_attribute_kernel,
    SOMAttributes *som_att = (SOMAttributes*)att;
    float *f_outputs = (float*)outputs;
    float rbf_scale = som_att->rbf_scale;

    float min = inputs[0];
    int min_node = 0;
    for (int i = 0 ; i < size ; ++i) {
        if (inputs[i] < min) {
            min = inputs[i];
            min_node = i;
        }
    }
    som_att->winner = min_node;

    ,

    float input = inputs[nid];
    SHIFT_FLOAT_OUTPUTS(f_outputs, std::exp(-rbf_scale * (input * input)));
)

/******************************************************************************/
/*************************** SOM ACTIVATOR KERNELS ****************************/
/******************************************************************************/

CALC_ALL(activate_som,
    ,

    float distance = 0.0;,

    float val = extract(outputs[from_index], delay) - weights[weight_index];
    distance += val * val;,

    inputs[to_index] += distance;
);

CALC_ALL(update_som,
    SOMAttributes *som_att = (SOMAttributes*)synapse_data.attributes;
    SOMWeightMatrix *som_mat = (SOMWeightMatrix*)synapse_data.matrix;
    float learning_rate = som_mat->learning_rate;
    float neighbor_learning_rate = som_mat->neighbor_learning_rate;
    int neighborhood_size = som_mat->neighborhood_size;
    int winner = som_att->winner;

    int winner_row = winner / to_columns;
    int winner_col = winner % to_columns;,

    int row_dist = abs(winner_row - (to_index / to_columns));
    int col_dist = abs(winner_col - (to_index % to_columns));,

    if (to_index == winner)
        weights[weight_index] += learning_rate *
            (extract(outputs[from_index], delay) - weights[weight_index]);
    else if (row_dist + col_dist <= neighborhood_size)
        weights[weight_index] += neighbor_learning_rate *
            (extract(outputs[from_index], delay) - weights[weight_index]);,

);

Kernel<SYNAPSE_ARGS> SOMAttributes::get_activator(Connection *conn) {
    try {
        if (not conn->second_order and conn->get_type() == FULLY_CONNECTED)
            return get_activate_som_fully_connected();
    } catch(std::out_of_range) { }

    LOG_ERROR(
        "Unimplemented connection type!");
}

/******************************************************************************/
/**************************** SOM UPDATER KERNELS *****************************/
/******************************************************************************/

Kernel<SYNAPSE_ARGS> SOMAttributes::get_updater(Connection *conn) {
    if (not conn->second_order and conn->get_type() == FULLY_CONNECTED)
        return get_update_som_fully_connected();
    else
        LOG_ERROR(
            "Unimplemented connection type!");
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

SOMAttributes::SOMAttributes(Layer *layer)
        : Attributes(layer, FLOAT) {
    this->winner = 0;
    this->rbf_scale = std::stof(layer->get_parameter("rbf scale", "1"));
}

void SOMAttributes::process_weight_matrix(WeightMatrix* matrix) {
    SOMWeightMatrix *s_mat = (SOMWeightMatrix*)matrix;
    auto conn = matrix->connection;

    s_mat->learning_rate =
        std::stof(conn->get_parameter("learning rate", "0.01"));
    s_mat->neighbor_learning_rate =
        std::stof(conn->get_parameter("neighbor learning rate", "0.001"));
    s_mat->neighborhood_size =
        std::stoi(conn->get_parameter("neighborhood size", "2"));
}
