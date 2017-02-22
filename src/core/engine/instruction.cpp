#include <cstring>
#include "engine/instruction.h"
#include "util/error_manager.h"

Instruction::Instruction(Layer *layer) : to_layer(layer) {
#ifdef PARALLEL
    // Default stream
    this->stream = 0;

    // Calculate grid and block sizes
    this->activator_threads = dim3(calc_threads(layer->size));
    this->activator_blocks = dim3(calc_blocks(layer->size));
#endif
}

#ifdef PARALLEL
void Instruction::record_events() {
    // Record added events
    for (int i = 0; i < this->events.size(); ++i)
        cudaEventRecord(this->events[i], this->stream);
}
#endif

InitializeInstruction::InitializeInstruction(Layer *layer, State *state)
        : Instruction(layer) {
    this->dst = state->get_input(layer);
}

void ClearInstruction::activate() {
#ifdef PARALLEL
    // Launch computation on provided stream, or default if none
    clear_data
        <<<activator_blocks, activator_threads, 0, this->stream>>>
        (dst, to_layer->size);
    Instruction::record_events();
#else
    clear_data(dst, to_layer->size);
#endif
}

void NoiseInstruction::activate() {
#ifdef PARALLEL
    // Launch computation on provided stream, or default if none
    randomize_data
        <<<activator_blocks, activator_threads, 0, this->stream>>>
        (dst, to_layer->size, to_layer->noise, init);
    Instruction::record_events();
#else
    randomize_data(dst, to_layer->size, to_layer->noise, init);
#endif
}

SynapseInstruction::SynapseInstruction(Connection *conn, State *state) :
        Instruction(conn->to_layer),
        connection(conn),
        synapse_data(conn, state),
        type(conn->type) {
    this->activator = state->get_activator(conn);
    this->updater = (conn->plastic) ? state->get_updater(conn) : NULL;

#ifdef PARALLEL
    if (conn->convolutional) {
        int num_weights = connection->get_num_weights();
        this->updater_threads = dim3(calc_threads(num_weights));
        this->updater_blocks = dim3(calc_blocks(num_weights));
    } else {
        this->updater_threads = dim3(calc_threads(to_layer->size));
        this->updater_blocks = dim3(calc_blocks(to_layer->size));
    }
#endif
}

void SynapseInstruction::activate() {
#ifdef PARALLEL
    // Launch computation on provided stream, or default if none
    activator
        <<<activator_blocks, activator_threads, 0, this->stream>>>
        (this->synapse_data);
    Instruction::record_events();
#else
    activator(this->synapse_data);
#endif
}

void SynapseInstruction::update() {
    if (this->updater != NULL) {
#ifdef PARALLEL
        updater
            <<<updater_blocks, updater_threads, 0, this->stream>>>
            (this->synapse_data);
#else
        updater(this->synapse_data);
#endif
    }
}

DendriticInstruction::DendriticInstruction(DendriticNode *parent,
    DendriticNode *child, State *state)
        : Instruction(parent->to_layer),
          init(child->register_index != 0) {
    int num_neurons = state->get_num_neurons(parent->to_layer->structure);
    float *input = state->get_input(to_layer);
    this->src = input + (num_neurons * child->register_index);
    this->dst = input + (num_neurons * parent->register_index);
}

void DendriticInstruction::activate() {
#ifdef PARALLEL
    // Launch computation on provided stream, or default if none
    calc_internal
        <<<activator_blocks, activator_threads, 0, this->stream>>>
        (to_layer->size, src, dst, this->init);
    Instruction::record_events();
#else
    calc_internal(to_layer->size, src, dst, this->init);
#endif
}

InputTransferInstruction::InputTransferInstruction(Layer *layer, State *state,
        Environment *environment) : Instruction(layer) {
    this->src = environment->get_buffer(layer->structure)->get_input(layer);
    this->dst = state->get_input(layer);
}

void InputTransferInstruction::activate() {
#ifdef PARALLEL
    // Copy to GPU from local location
    cudaMemcpyAsync(dst, src, to_layer->size * sizeof(float),
        cudaMemcpyHostToDevice, this->stream);
    cudaCheckError("Failed to transfer input!");
    Instruction::record_events();
#else
    memcpy(dst, src, to_layer->size * sizeof(float));
#endif
}

OutputTransferInstruction::OutputTransferInstruction(Layer *layer, State *state,
        Environment *environment) : Instruction(layer) {
    this->src = state->get_output(layer);
    this->dst = environment->get_buffer(layer->structure)->get_output(layer);
}

void OutputTransferInstruction::activate() {
#ifdef PARALLEL
    // Copy to GPU from local location
    cudaMemcpyAsync(dst, src, to_layer->size * sizeof(Output),
        cudaMemcpyDeviceToHost, this->stream);
    cudaCheckError("Failed to transfer output!");
    Instruction::record_events();
#else
    memcpy(dst, src, to_layer->size * sizeof(Output));
#endif
}

void StateUpdateInstruction::activate() {
#ifdef PARALLEL
    attribute_kernel<<<activator_blocks, activator_threads, 0, this->stream>>>(
        attributes, start_index, to_layer->size);
    cudaCheckError("Failed to update neuron state/output!");
    Instruction::record_events();
#else
    attribute_kernel(attributes, start_index, to_layer->size);
#endif
}
