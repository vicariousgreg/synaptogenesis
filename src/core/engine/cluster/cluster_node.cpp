#include "engine/cluster/cluster_node.h"
#include "network/layer.h"
#include "engine/engine.h"
#include "engine/instruction.h"
#include "util/error_manager.h"

ClusterNode::ClusterNode(Layer *layer, State *state, Engine *engine,
        Stream *io_stream, Stream *compute_stream)
        : to_layer(layer),
          device_id(state->get_device_id(layer)),
          is_input(engine->is_input(layer)),
          is_expected(engine->is_expected(layer)),
          is_output(engine->is_output(layer)),
          input_instruction(nullptr),
          expected_instruction(nullptr),
          output_instruction(nullptr),
          state_update_instruction(nullptr),
          state_learning_instruction(nullptr),
          io_stream(io_stream),
          compute_stream(compute_stream),
          state(state),
          engine(engine) {
    // Add input transfer instruction
    if (this->is_input)
        this->input_instruction =
            new InputTransferInstruction(
                to_layer, state, engine, compute_stream);

    if (this->is_expected)
        this->expected_instruction =
            new ExpectedTransferInstruction(
                to_layer, state, engine, compute_stream);

    // Add noise / clear instruction
    auto noise_config = to_layer->get_config()->get_child("noise config", nullptr);
    if (noise_config != nullptr) {
        auto type = noise_config->get("type", "normal");
        if (type == "normal")
            activate_instructions.push_back(
                new NormalNoiseInstruction(
                    to_layer, state, compute_stream,
                    not engine->is_input(to_layer)));
        else if (type == "poisson")
            activate_instructions.push_back(
                new PoissonNoiseInstruction(
                    to_layer, state, compute_stream,
                    not engine->is_input(to_layer)));
        else
            ErrorManager::get_instance()->log_error(
                "Error building cluster node for " + layer->str() + ":\n"
                "  Unrecognized noise type: " + type);
    } else if (not this->is_input) {
        activate_instructions.push_back(
            new ClearInstruction(to_layer, state, compute_stream));
    }

    // Add state instructions
    this->state_update_instruction =
        new StateUpdateInstruction(
            to_layer, state, compute_stream);
    if (layer->plastic and
            not state->get_learning_kernel(layer).is_null()) {
        this->state_learning_instruction =
            new StateLearningInstruction(
                to_layer, state, compute_stream);
        update_instructions.push_back(this->state_learning_instruction);
    }

    // Perform DFS on dendritic tree
    // Do this after so that the state learning instruction comes first
    //   in the update_instructions list
    dendrite_DFS(to_layer->get_dendritic_root());

    // Add output transfer instruction
    if (this->is_output) {
        this->output_instruction =
            new OutputTransferInstruction(
                to_layer, state, engine, io_stream);

        // Ensure output and state instructions depend on one another
        output_instruction->add_dependency(state_update_instruction);
        state_update_instruction->add_dependency(output_instruction);
    }
}

ClusterNode::~ClusterNode() {
    for (auto& inst : this->activate_instructions) delete inst;
    for (auto& inst : this->update_instructions) delete inst;
    if (is_input) delete input_instruction;
    if (is_expected) delete expected_instruction;
    if (is_output) delete output_instruction;
    delete state_update_instruction;
    if (state_learning_instruction != nullptr)
        delete state_learning_instruction;
}

void ClusterNode::dendrite_DFS(DendriticNode *curr) {
    // Second order connections need a transfer to copy the host weights
    if (curr->second_order)
        activate_instructions.push_back(
            new SecondOrderWeightTransferInstruction(
                curr, state, compute_stream));

    for (auto& child : curr->get_children()) {
        // Create an instruction
        // If internal, recurse first (post-fix DFS)
        if (child->is_leaf()) {
            Connection *conn = child->conn;

            auto syn_inst = new SynapseActivateInstruction(
                curr, conn, state, compute_stream);

            // Create the instruction and add it to the synapse instuction list
            synapse_activate_instructions.push_back(syn_inst);
            activate_instructions.push_back(syn_inst);

            // If plastic, create update instruction
            if (conn->plastic) {
                auto syn_update_inst = new SynapseUpdateInstruction(
                    curr, conn, state, compute_stream);
                update_instructions.push_back(syn_update_inst);
                synapse_update_instructions.push_back(syn_update_inst);
            }
        } else {
            this->dendrite_DFS(child);

            // If the registers do not match, add a transfer instruction
            // Second order connections won't reach here
            //   because they can't have internal children
            if (curr->register_index != child->register_index)
                activate_instructions.push_back(
                    new DendriticInstruction(
                        curr, child, state, compute_stream));
        }
    }

    // Add second order host connection if applicable
    // The host connection is deferred to the end because other non-host
    //   connections operate on a copy of its weight matrix
    // The host connection will use this copy during its synaptic
    //   computations (see synapse_data.cpp)
    if (curr->second_order) {
        Connection *conn = curr->get_second_order_connection();

        if (conn == nullptr)
            ErrorManager::get_instance()->log_error(
                "Error building cluster node for " + curr->to_layer->str() + ":\n"
                "  Missing host connection for second order node!");

        auto syn_inst = new SynapseActivateInstruction(
            curr, conn, state, compute_stream);

        // Create the instruction and add it to the synapse instuction list
        synapse_activate_instructions.push_back(syn_inst);
        activate_instructions.push_back(syn_inst);

        // If plastic, create update instruction
        if (conn->plastic) {
            auto syn_update_inst = new SynapseUpdateInstruction(
                curr, conn, state, compute_stream);
            update_instructions.push_back(syn_update_inst);
            synapse_update_instructions.push_back(syn_update_inst);
        }
    }
}

void ClusterNode::activate_input() {
    if (this->is_input) input_instruction->activate();
    if (this->is_expected) expected_instruction->activate();
}

void ClusterNode::activate_state() {
    state_update_instruction->activate();
}

void ClusterNode::activate_output() {
    if (this->is_output) output_instruction->activate();
}

void ClusterNode::synchronize_input() {
    if (is_input)    input_instruction->synchronize();
    if (is_expected) expected_instruction->synchronize();
}

void ClusterNode::synchronize_output() {
    if (is_output) output_instruction->synchronize();
}

const InstructionList& ClusterNode::get_activate_instructions() const {
    return activate_instructions;
}

const InstructionList& ClusterNode::get_update_instructions() const {
    return update_instructions;
}

Instruction* ClusterNode::get_input_instruction() const {
    return input_instruction;
}

Instruction* ClusterNode::get_state_update_instruction() const {
    return state_update_instruction;
}

Instruction* ClusterNode::get_output_instruction() const {
    return output_instruction;
}

const SynapseInstructionList& ClusterNode::get_synapse_activate_instructions() const {
    return synapse_activate_instructions;
}

const SynapseInstructionList& ClusterNode::get_synapse_update_instructions() const {
    return synapse_update_instructions;
}
