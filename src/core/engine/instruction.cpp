#include "engine/instruction.h"
#include "util/error_manager.h"

SynapseInstruction::SynapseInstruction(Connection *conn, State *state) :
        connection(conn),
        kernel_data(conn, state),
        type(conn->type) {
    this->activator = state->get_activator(type);
    this->updater = (conn->plastic) ? state->get_updater(type) : NULL;

#ifdef PARALLEL
    this->stream = NULL;

    // Calculate grid and block sizes
    this->activator_threads = dim3(calc_threads(conn->to_layer->size));
    this->activator_blocks = dim3(calc_blocks(conn->to_layer->size));

    if (conn->convolutional) {
        int num_weights = connection->get_num_weights();
        this->updater_threads = dim3(calc_threads(num_weights));
        this->updater_blocks = dim3(calc_blocks(num_weights));
    } else {
        this->updater_threads = dim3(calc_threads(conn->to_layer->size));
        this->updater_blocks = dim3(calc_blocks(conn->to_layer->size));
    }
#endif
}

void SynapseInstruction::activate() {
#ifdef PARALLEL
    // Launch computation on provided stream, or default if none
    if (this->stream) {
        activator
            <<<activator_blocks, activator_threads, 0, *this->stream>>>
            (this->kernel_data);

        // Record added events
        for (int i = 0; i < this->events.size(); ++i)
            cudaEventRecord(*this->events[i], *this->stream);
    } else {
        activator
            <<<activator_blocks, activator_threads>>>
            (this->kernel_data);

        // Record added events
        for (int i = 0; i < this->events.size(); ++i)
            cudaEventRecord(*this->events[i]);
    }
#else
    activator(this->kernel_data);
#endif
}

void SynapseInstruction::update() {
    if (this->kernel_data.plastic) {
#ifdef PARALLEL
        if (this->stream)
            updater
                <<<updater_blocks, updater_threads, 0, *this->stream>>>
                (this->kernel_data);
        else
            updater
                <<<updater_blocks, updater_threads>>>
                (this->kernel_data);
#else
        updater(this->kernel_data);
#endif
    }
}

DendriticInstruction::DendriticInstruction(DendriticNode *parent,
    DendriticNode *child, State *state)
        : to_layer(parent->to_layer),
          size(to_layer->size),
          clear(child->register_index != 0) {
    int num_neurons = state->get_num_neurons();
    this->src =
        state->get_input()
        + (num_neurons * child->register_index)
        + to_layer->get_start_index();
    this->dst =
        state->get_input()
        + (num_neurons * parent->register_index)
        + to_layer->get_start_index();

#ifdef PARALLEL
    this->stream = NULL;

    // Calculate grid and block sizes
    this->activator_threads = dim3(calc_threads(to_layer->size));
    this->activator_blocks = dim3(calc_blocks(to_layer->size));
#endif
}

void DendriticInstruction::activate() {
#ifdef PARALLEL
    // Launch computation on provided stream, or default if none
    if (this->stream)
        calc_internal
            <<<activator_blocks, activator_threads, 0, *this->stream>>>
            (size, src, dst, this->clear);
    else
        calc_internal
            <<<activator_blocks, activator_threads>>>
            (size, src, dst, this->clear);

    // Record added events
    for (int i = 0; i < this->events.size(); ++i)
        cudaEventRecord(*this->events[i], *this->stream);
#else
    calc_internal(size, src, dst, this->clear);
#endif
}
