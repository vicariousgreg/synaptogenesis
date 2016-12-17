#include "engine/instruction.h"
#include "util/error_manager.h"

Instruction::Instruction(Connection *conn, State *state) :
        connection(conn),
        kernel_data(conn, state),
        type(conn->type) {
    get_activator_kernel(&this->activator, type);
    if (conn->plastic)
        get_updater_kernel(&this->updater, type);
    else
        this->updater = NULL;

#ifdef PARALLEL
    this->stream = NULL;

    // Calculate grid and block sizes based on type
    int threads = calc_threads(conn->to_layer->size);

    switch (type) {
        case FULLY_CONNECTED:
        case ONE_TO_ONE:
            this->blocks_per_grid = dim3(calc_blocks(conn->to_layer->size));
            this->threads_per_block = dim3(threads);
            break;
        case CONVERGENT:
        case CONVOLUTIONAL:
            this->blocks_per_grid = dim3(
                conn->to_layer->rows,
                calc_blocks(conn->to_layer->columns));
            this->threads_per_block = dim3(1, threads);
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
#endif
}

void Instruction::disable_learning() {
    this->kernel_data.plastic = false;
}

void Instruction::activate() {
#ifdef PARALLEL
    // Launch computation on provided stream, or default if none
    if (this->stream)
        activator
            <<<blocks_per_grid, threads_per_block, 0, *this->stream>>>
            (this->kernel_data);
    else
        activator
            <<<blocks_per_grid, threads_per_block>>>
            (this->kernel_data);

    // Record added events
    for (int i = 0; i < this->events.size(); ++i)
        cudaEventRecord(*this->events[i], *this->stream);
#else
    activator(this->kernel_data);
#endif
}

void Instruction::update() {
    if (this->kernel_data.plastic)
#ifdef PARALLEL
        if (this->stream)
            updater
                <<<blocks_per_grid, threads_per_block, 0, *this->stream>>>
                (this->kernel_data);
        else
            updater
                <<<blocks_per_grid, threads_per_block>>>
                (this->kernel_data);
#else
        updater(this->kernel_data);
#endif
}
