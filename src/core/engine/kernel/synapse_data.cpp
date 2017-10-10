#include "engine/kernel/synapse_data.h"
#include "state/state.h"
#include "state/attributes.h"

SynapseData::SynapseData(DendriticNode *parent_node,
    Connection *conn, State *state) :
        attributes(state->get_attributes_pointer(conn->to_layer)),
        extractor(state->get_connection_extractor(conn)),
        convolutional(conn->convolutional),
        connection_index(state->get_connection_index(conn)),
        opcode(conn->opcode),
        plastic(conn->plastic),
        max_weight(conn->max_weight),
        subset_config(conn->get_config()->get_subset_config()),
        arborized_config(conn->get_config()->get_arborized_config()),
        delay(conn->delay),
        from_size(conn->from_layer->size),
        from_rows(conn->from_layer->rows),
        from_columns(conn->from_layer->columns),
        to_size(conn->to_layer->size),
        to_rows(conn->to_layer->rows),
        to_columns(conn->to_layer->columns),
        to_start_index(state->get_other_start_index(conn->to_layer)),
        to_layer_index(state->get_layer_index(conn->to_layer)),
        num_weights(conn->get_num_weights()),
        output_type(Attributes::get_output_type(conn->from_layer)),
        weights(state->get_matrix(conn)) {
    destination_outputs = state->get_output(conn->to_layer);

    if (state->is_inter_device(conn))
        outputs = state->get_device_output_buffer(conn,
            get_word_index(conn->delay, output_type));
    else
        outputs = state->get_output(conn->from_layer,
            get_word_index(conn->delay, output_type));
    inputs = state->get_input(conn->to_layer, parent_node->register_index);

    // Second order non-host connections operate on copies of the
    //   host connection's weight matrix
    if (conn->second_order)
        second_order_weights = state->get_second_order_weights(parent_node);

    // The second order host uses the operated upon copy as its source matrix
    // TODO: Figure out a better way to deal with this for stacked matrices
    if (conn->second_order_host)
        weights = second_order_weights;
}
