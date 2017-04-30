#include "engine/cluster/cluster_node.h"
#include "model/layer.h"
#include "io/environment.h"
#include "engine/instruction.h"
#include "util/error_manager.h"

ClusterNode::ClusterNode(Layer *layer, State *state, Environment *environment,
        Stream *io_stream, Stream *compute_stream)
        : to_layer(layer),
          device_id(state->get_device_id(layer)),
          is_input(layer->is_input()),
          is_expected(layer->is_expected()),
          is_output(layer->is_output()),
          input_instruction(nullptr),
          expected_instruction(nullptr),
          output_instruction(nullptr),
          state_update_instruction(nullptr),
          state_learning_instruction(nullptr),
          io_stream(io_stream),
          compute_stream(compute_stream),
          state(state),
          environment(environment) {
    auto res_man = ResourceManager::get_instance();

    // Add input transfer instruction
    if (this->is_input)
        this->input_instruction =
            new InputTransferInstruction(
                to_layer, state, environment, compute_stream);

    if (this->is_expected)
        this->expected_instruction =
            new ExpectedTransferInstruction(
                to_layer, state, environment, compute_stream);

    // Add noise / clear instruction
    if (to_layer->noise_mean != 0.0)
        activate_instructions.push_back(
            new NoiseInstruction(to_layer, state, compute_stream));
    else if (not this->is_input)
        activate_instructions.push_back(
            new ClearInstruction(to_layer, state, compute_stream));

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
    dendrite_DFS(to_layer->dendritic_root);

    // Add output transfer instruction
    if (this->is_output)
        this->output_instruction =
            new OutputTransferInstruction(
                to_layer, state, environment, io_stream);
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
    auto res_man = ResourceManager::get_instance();

    // Second order connections need their own clear instruction
    if (curr->is_second_order())
        activate_instructions.push_back(
            new ClearInstruction(curr, state, compute_stream));

    // Keep track of consecutive subset connections
    // If they come from the same layer, only wrap the first one
    //   in an inter-device instruction (if necessary)
    bool in_subset_chain = false;
    Layer *prev_from_layer = nullptr;

    for (auto& child : curr->get_children()) {
        // Create an instruction
        // If internal, recurse first (post-fix DFS)
        if (child->is_leaf()) {
            Connection *conn = child->conn;

            Instruction* syn_inst = new SynapseActivateInstruction(
                curr, conn, state, compute_stream);

            // If inter-device, wrap in InterDeviceInstruction
            if (not in_subset_chain and state->is_inter_device(conn))
                syn_inst = new InterDeviceInstruction(
                    conn, state, syn_inst);

            // Only continue the chain if the from_layer remains the same
            // If this is a subset connection
            //     We are in a chain if we weren't before or if the
            //         previous from_layer has not changed
            // Otherwise, we are not in a chain
            in_subset_chain =
                (conn->type == SUBSET and
                    (not in_subset_chain or conn->from_layer == prev_from_layer));
            prev_from_layer = conn->from_layer;

            // Create the instruction and add it to the synapse instuction list
            synapse_instructions[conn] = syn_inst;
            activate_instructions.push_back(syn_inst);

            // If plastic, create update instruction
            if (conn->plastic) {
                auto syn_update_inst = new SynapseUpdateInstruction(
                    curr, conn, state, compute_stream);
                update_instructions.push_back(syn_update_inst);
            }
        } else {
            this->dendrite_DFS(child);

            // Second order connections won't reach here
            //   because they can't have internal children
            activate_instructions.push_back(
                new DendriticInstruction(
                    curr, child, state, compute_stream));
        }
    }

    // Add second order aggregator if applicable
    if (curr->is_second_order())
        activate_instructions.push_back(
            new SecondOrderDendriticInstruction(
                curr, state, compute_stream));
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
    if (is_input) input_instruction->synchronize();
    if (is_expected) expected_instruction->synchronize();
}

void ClusterNode::synchronize_output() {
    if (is_output) output_instruction->synchronize();
}

const InstructionList ClusterNode::get_activate_instructions() const {
    return activate_instructions;
}

const InstructionList ClusterNode::get_update_instructions() const {
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

const std::map<Connection*, Instruction*>
        ClusterNode::get_synapse_instructions() const {
    return synapse_instructions;
}
