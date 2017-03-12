#include "engine/kernel/synapse_data.h"
#include "state/state.h"
#include "state/attributes.h"
#include "util/error_manager.h"

SynapseData::SynapseData(Connection *conn, State *state) :
        extractor(state->get_extractor(conn)),
        convolutional(conn->convolutional),
        opcode(conn->opcode),
        plastic(conn->plastic),
        max_weight(conn->max_weight),
        field_size(conn->get_field_size()),
        stride(conn->get_stride()),
        delay(conn->delay),
        from_size(conn->from_layer->size),
        from_rows(conn->from_layer->rows),
        from_columns(conn->from_layer->columns),
        to_size(conn->to_layer->size),
        to_rows(conn->to_layer->rows),
        to_columns(conn->to_layer->columns),
        num_weights(conn->get_num_weights()),
        output_type(state->get_output_type(conn->to_layer)),
        weights(state->get_matrix(conn)) {
    this->fray =
        (to_rows == from_rows and to_columns == from_columns)
            ? field_size / 2 : 0;

    destination_outputs = state->get_output(conn->to_layer);
    if (state->is_inter_device(conn))
        outputs = state->get_device_output_buffer(conn);
    else
        outputs = state->get_output(conn->from_layer,
            get_word_index(conn->delay, output_type));
    inputs = state->get_input(conn->to_layer);
}
