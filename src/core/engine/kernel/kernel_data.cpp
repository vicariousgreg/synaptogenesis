#include "engine/kernel/kernel_data.h"
#include "util/error_manager.h"

KernelData::KernelData(Connection *conn, State *state) :
        convolutional(conn->convolutional),
        opcode(conn->opcode),
        plastic(conn->plastic),
        max_weight(conn->max_weight),
        overlap(conn->overlap),
        stride(conn->stride),
        delay(conn->delay),
        from_size(conn->from_layer->size),
        from_rows(conn->from_layer->rows),
        from_columns(conn->from_layer->columns),
        to_size(conn->to_layer->size),
        to_rows(conn->to_layer->rows),
        to_columns(conn->to_layer->columns),
        num_weights(conn->num_weights),
        output_type(state->get_attributes()->get_output_type()),
        weights(state->get_matrix(conn->id)) {
    this->fray =
        (to_rows == from_rows and to_columns == from_columns)
            ? overlap / 2 : 0;

    // Set up word index
    int timesteps_per_output = get_timesteps_per_output(output_type);
    int word_index = HISTORY_SIZE - 1 -
        (conn->delay / timesteps_per_output);
    if (word_index < 0)
        ErrorManager::get_instance()->log_error(
            "Invalid delay in connection!");

    outputs = state->get_attributes()->get_output(word_index)
                + conn->from_layer->index;
    inputs = state->get_attributes()->get_input()
                + conn->to_layer->index;

    get_extractor(&this->extractor, output_type);
}
