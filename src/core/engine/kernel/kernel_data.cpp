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

/******************************************************************************/
/************************** OUTPUT EXTRACTORS *********************************/
/******************************************************************************/

// Device pointers for memcpyFromSymbol
DEVICE EXTRACTOR x_float = extract_float;
DEVICE EXTRACTOR x_int = extract_int;
DEVICE EXTRACTOR x_bit = extract_bit;

DEVICE float extract_float(KernelData &kernel_data, Output &out) { return out.f; }
DEVICE float extract_int(KernelData &kernel_data, Output &out) { return out.i; }
DEVICE float extract_bit(KernelData &kernel_data, Output &out) {
    return (out.i >> (kernel_data.delay % 32)) & 1;
}

void get_extractor(EXTRACTOR *dest, OutputType output_type) {
#ifdef PARALLEL
    switch (output_type) {
        case FLOAT:
            cudaMemcpyFromSymbol(dest, x_float, sizeof(void *));
            break;
        case INT:
            cudaMemcpyFromSymbol(dest, x_int, sizeof(void *));
            break;
        case BIT:
            cudaMemcpyFromSymbol(dest, x_bit, sizeof(void *));
            break;
    }
#else
    switch (output_type) {
        case FLOAT:
            *dest = extract_float;
            break;
        case INT:
            *dest = extract_int;
            break;
        case BIT:
            *dest = extract_bit;
            break;
    }
#endif
}
