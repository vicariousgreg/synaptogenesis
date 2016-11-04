#include "engine/instruction.h"
#include "util/error_manager.h"

Instruction::Instruction(Connection *conn, State *state) :
        type(conn->type),
        convolutional(conn->convolutional),
        opcode(conn->opcode),
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
        inputs(inputs + conn->to_layer->index),
        weights(state->get_matrix(conn->id)) {

    // Set up word index
    int timesteps_per_output = get_timesteps_per_output(output_type);
    int word_index = HISTORY_SIZE - 1 -
        (conn->delay / timesteps_per_output);
    if (word_index < 0)
        ErrorManager::get_instance()->log_error(
            "Invalid delay in connection!");

    outputs = state->get_attributes()->get_output(word_index) + conn->from_layer->index,
    inputs = state->get_attributes()->get_input() + conn->to_layer->index;

    this->fray = 
        (to_rows == from_rows and to_columns == from_columns)
            ? overlap / 2 : 0;

    get_kernel(&this->kernel, type);
    get_extractor(&this->extractor, output_type);

#ifdef PARALLEL
    // Calculate grid and block sizes based on type
    int threads = calc_threads(to_size);

    switch (type) {
        case (FULLY_CONNECTED):
        case (ONE_TO_ONE):
            blocks_per_grid = dim3(calc_blocks(to_size));
            threads_per_block = dim3(threads);
            break;
        case (CONVERGENT):
        case (CONVERGENT_CONVOLUTIONAL):
            blocks_per_grid = dim3(
                to_rows,
                calc_blocks(to_columns));
            threads_per_block = dim3(1, threads);
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
#endif
}

#ifdef PARALLEL
void Instruction::execute(cudaStream_t *stream) {
    kernel<<<blocks_per_grid, threads_per_block, 0, *stream>>>(*this);
}
#else
void Instruction::execute() {
    kernel(*this);
}
#endif
