#include <cstring>
#include "engine/stream/instruction.h"
#include "util/error_manager.h"

Instruction::Instruction(Layer *layer) : to_layer(layer) {
    // Calculate grid and block sizes
    this->activator_threads = calc_threads(layer->size);
    this->activator_blocks = calc_blocks(layer->size);

    // Default stream
    this->stream = Stream::get_default_stream();
}

void Instruction::record_events() {
    // Record added events
    for (auto& event : events) stream->record(event);
}

InitializeInstruction::InitializeInstruction(Layer *layer, State *state)
        : Instruction(layer) {
    this->dst = state->get_input(layer);
}

void ClearInstruction::activate() {
    stream->run_kernel(clear_data, activator_blocks, activator_threads,
        dst, to_layer->size);
    Instruction::record_events();
}

void NoiseInstruction::activate() {
    stream->run_kernel(randomize_data, activator_blocks, activator_threads,
        dst, to_layer->size, to_layer->noise, init);
    Instruction::record_events();
}

SynapseInstruction::SynapseInstruction(Connection *conn, State *state) :
        Instruction(conn->to_layer),
        connection(conn),
        synapse_data(conn, state),
        type(conn->type) {
    this->activator = state->get_activator(conn);
    this->updater = (conn->plastic) ? state->get_updater(conn) : NULL;

    if (conn->convolutional) {
        int num_weights = connection->get_num_weights();
        this->updater_threads = calc_threads(num_weights);
        this->updater_blocks = calc_blocks(num_weights);
    } else {
        this->updater_threads = calc_threads(to_layer->size);
        this->updater_blocks = calc_blocks(to_layer->size);
    }
}

void SynapseInstruction::activate() {
    stream->run_kernel(activator, activator_blocks, activator_threads,
        this->synapse_data);
    Instruction::record_events();
}

void SynapseInstruction::update() {
    if (this->updater != NULL)
        stream->run_kernel(updater, updater_blocks, updater_threads,
            this->synapse_data);
}

DendriticInstruction::DendriticInstruction(DendriticNode *parent,
    DendriticNode *child, State *state)
        : Instruction(parent->to_layer),
          init(child->register_index != 0) {
    this->src = state->get_input(to_layer, child->register_index);
    this->dst = state->get_input(to_layer, parent->register_index);
}

void DendriticInstruction::activate() {
    stream->run_kernel(calc_internal, activator_blocks, activator_threads,
        to_layer->size, src, dst, this->init);
    Instruction::record_events();
}

InputTransferInstruction::InputTransferInstruction(Layer *layer, State *state,
        Environment *environment) : Instruction(layer) {
    this->src = environment->buffer->get_input(layer);
    this->dst = state->get_input(layer);
}

void InputTransferInstruction::activate() {
    src.copy_to(dst);
    Instruction::record_events();
}

OutputTransferInstruction::OutputTransferInstruction(Layer *layer, State *state,
        Environment *environment) : Instruction(layer) {
    this->src = state->get_output(layer);
    this->dst = environment->buffer->get_output(layer);
}

void OutputTransferInstruction::activate() {
    src.copy_to(dst);
    Instruction::record_events();
}

void StateUpdateInstruction::activate() {
    stream->run_kernel(attribute_kernel, activator_blocks, activator_threads,
        attribute_data);
    Instruction::record_events();
}
