#include "engine/kernel/synapse_data.h"
#include "state/state.h"
#include "state/attributes.h"

SynapseData::SynapseData(DendriticNode *parent_node,
    Connection *conn, State *state) :
        to_attributes(state->get_attributes_pointer(conn->to_layer)),
        extractor(state->get_extractor(conn)),
        convolutional(conn->convolutional),
        connection_index(state->get_connection_index(conn)),
        opcode(conn->opcode),
        plastic(conn->plastic),
        max_weight(conn->max_weight),
        fully_connected_config(conn->get_config()->copy_fully_connected_config()),
        arborized_config(conn->get_config()->copy_arborized_config()),
        delay(conn->delay),
        second_order(parent_node->is_second_order()),
        from_size(conn->from_layer->size),
        from_rows(conn->from_layer->rows),
        from_columns(conn->from_layer->columns),
        to_size(conn->to_layer->size),
        to_rows(conn->to_layer->rows),
        to_columns(conn->to_layer->columns),
        to_start_index(state->get_other_start_index(conn->to_layer)),
        to_layer_index(state->get_layer_index(conn->to_layer)),
        num_weights(conn->get_num_weights()),
        output_type(state->get_output_type(conn->from_layer)),
        weights(state->get_matrix(conn)) {
    destination_outputs = state->get_output(conn->to_layer);
    if (state->is_inter_device(conn))
        outputs = state->get_device_output_buffer(conn);
    else
        outputs = state->get_output(conn->from_layer,
            get_word_index(conn->delay, output_type));
    inputs = state->get_input(conn->to_layer);

    if (second_order)
        second_order_inputs = state->get_second_order_input(parent_node);
}
