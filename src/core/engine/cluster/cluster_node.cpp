#include "engine/cluster/cluster_node.h"
#include "network/layer.h"
#include "engine/engine.h"
#include "engine/instruction.h"
#include "util/logger.h"

ClusterNode::ClusterNode(Layer *layer, State *state, Engine *engine,
        Stream *io_stream, Stream *compute_stream)
        : to_layer(layer),
          device_id(state->get_device_id(layer)),
          input_instruction(nullptr),
          output_instruction(nullptr),
          state_update_instruction(nullptr),
          state_learning_instruction(nullptr),
          io_stream(io_stream),
          compute_stream(compute_stream),
          state(state),
          engine(engine) {
    // Add input transfer instruction
    if (engine->is_input(layer)) {
        auto keys = engine->get_input_keys(layer);

        // Add auxiliary input instructions
        for (auto key : keys) {
            LOG_DEBUG("Adding input key " + key +
                " for layer " + layer->str() + "\n");

            if (key == "input")
                this->input_instruction =
                    new InputTransferInstruction(
                        to_layer, state, engine, compute_stream);
            else
                input_auxiliary_instructions.push_back(
                    new InputAuxiliaryTransferInstruction(key,
                        to_layer, state, engine, compute_stream));
        }
    }

    // Add init (init / clear) instruction
    Instruction *init_instruction = get_initialize_instruction(
        to_layer, state, compute_stream, input_instruction != nullptr);
    if (init_instruction != nullptr)
        activate_instructions.push_back(init_instruction);

    // Add any custom initialization instructions
    for (auto key : state->get_init_keys(to_layer))
        this->activate_instructions.push_back(
            new AuxiliaryInitializeInstruction(
                to_layer, state, compute_stream, key, 0.0, true));

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
    if (engine->is_output(layer)) {
        auto keys = engine->get_output_keys(layer);

        // Add auxiliary output instructions
        for (auto key : keys) {
            LOG_DEBUG("Adding output key " + key +
                " for layer " + layer->str() + "\n");

            Instruction *inst = nullptr;

            if (key == "output") {
                inst = this->output_instruction =
                    new OutputTransferInstruction(
                        to_layer, state, engine, io_stream);
            } else {
                inst = new OutputAuxiliaryTransferInstruction(key,
                    to_layer, state, engine, io_stream);
                output_auxiliary_instructions.push_back(inst);
            }

            // Ensure output and state instructions depend on one another
            inst->add_dependency(state_update_instruction);
            state_update_instruction->add_dependency(inst);
        }
    }
}

ClusterNode::~ClusterNode() {
    for (auto& inst : this->activate_instructions) delete inst;
    for (auto& inst : this->update_instructions) delete inst;
    if (input_instruction != nullptr) delete input_instruction;
    if (output_instruction != nullptr) delete output_instruction;
    delete state_update_instruction;
    if (state_learning_instruction != nullptr)
        delete state_learning_instruction;
    for (auto& inst : input_auxiliary_instructions) delete inst;
    for (auto& inst : output_auxiliary_instructions) delete inst;
}

void ClusterNode::dendrite_DFS(DendriticNode *curr) {
    // Second order connections need a transfer to copy the host weights
    if (curr->second_order)
        activate_instructions.push_back(
            new SecondOrderWeightTransferInstruction(
                curr, state, compute_stream));
    // During propagation up the dendritic tree, internal nodes will be set to
    //   their initialization value upon exit.  Thus, we need to initialize the
    //   values for the first round.  Create a temporary instruction and do it.
    // This shouldn't cause problems because the stream will be held up until
    //   this initialization instruction is completed.
    // This doesn't get run for root because it cannot have an init_value, and
    //   because it can get init applied to it that should not be overwritten
    else if (curr->name != "root")
        (SetInstruction(curr, state, compute_stream, curr->init_val)).activate();

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

            // If transpose flag, add transposition instruction
            if (state->get_transpose_flag(conn))
                update_instructions.push_back(
                    new TransposeInstruction(conn, state, compute_stream));

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
            LOG_ERROR(
                "Error building cluster node for " + curr->to_layer->str() + ":\n"
                "  Missing host connection for second order node!");

        auto syn_inst = new SynapseActivateInstruction(
            curr, conn, state, compute_stream);

        // Create the instruction and add it to the synapse instuction list
        synapse_activate_instructions.push_back(syn_inst);
        activate_instructions.push_back(syn_inst);

        // If transpose flag, add transposition instruction
        if (state->get_transpose_flag(conn))
            update_instructions.push_back(
                new TransposeInstruction(conn, state, compute_stream));

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
    if (input_instruction != nullptr)
        input_instruction->activate();
    for (auto& inst : input_auxiliary_instructions)
        inst->activate();
}

void ClusterNode::activate_state() {
    state_update_instruction->activate();
}

void ClusterNode::activate_output() {
    if (output_instruction != nullptr)
        output_instruction->activate();
    for (auto& inst : output_auxiliary_instructions)
        inst->activate();
}

void ClusterNode::synchronize_input() {
    if (input_instruction != nullptr)
        input_instruction->synchronize();
    for (auto& inst : input_auxiliary_instructions)
        inst->synchronize();
}

void ClusterNode::synchronize_output() {
    if (output_instruction != nullptr)
        output_instruction->synchronize();
    for (auto& inst : output_auxiliary_instructions)
        inst->synchronize();
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
