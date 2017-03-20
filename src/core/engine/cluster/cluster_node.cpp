#include "engine/cluster/cluster_node.h"
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
          state_instruction(nullptr),
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
    if (to_layer->noise != 0.0)
        activate_instructions.push_back(
            new NoiseInstruction(to_layer, state, compute_stream));
    else if (not this->is_input)
        activate_instructions.push_back(
            new ClearInstruction(to_layer, state, compute_stream));

    // Beform DFS on dendritic tree
    dendrite_DFS(to_layer->dendritic_root);

    // Add state instruction
    this->state_instruction =
        new StateUpdateInstruction(
            to_layer, state, compute_stream);

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
    delete state_instruction;
}

void ClusterNode::dendrite_DFS(DendriticNode *curr) {
    auto res_man = ResourceManager::get_instance();

    for (auto& child : curr->get_children()) {
        // Create an instruction
        // If internal, recurse first (post-fix DFS)
        if (child->is_leaf()) {
            auto syn_inst = new SynapseActivateInstruction(
                child->conn, state, compute_stream);

            // Create the instruction and add it to the synapse instuction list
            synapse_instructions[child->conn] = syn_inst;
            activate_instructions.push_back(syn_inst);

            // If plastic, create update instruction
            if (child->conn->plastic) {
                auto syn_update_inst = new SynapseUpdateInstruction(
                    child->conn, state, compute_stream);
                update_instructions.push_back(syn_update_inst);
            }
        } else {
            this->dendrite_DFS(child);
            activate_instructions.push_back(
                new DendriticInstruction(
                    curr, child, state, compute_stream));
        }
    }
}

void ClusterNode::activate_input() {
    if (this->is_input) input_instruction->activate();
    if (this->is_expected) expected_instruction->activate();
}

void ClusterNode::activate_state() {
    state_instruction->activate();
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

Instruction* ClusterNode::get_state_instruction() const {
    return state_instruction;
}

Instruction* ClusterNode::get_output_instruction() const {
    return output_instruction;
}

const std::map<Connection*, Instruction*>
        ClusterNode::get_synapse_instructions() const {
    return synapse_instructions;
}
