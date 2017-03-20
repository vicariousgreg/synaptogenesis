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
          input_copy_instruction(nullptr),
          expected_instruction(nullptr),
          expected_copy_instruction(nullptr),
          output_instruction(nullptr),
          output_copy_instruction(nullptr),
          state_instruction(nullptr),
          io_stream(io_stream),
          compute_stream(compute_stream),
          state(state),
          environment(environment) {
    auto res_man = ResourceManager::get_instance();

    // Add input transfer instruction
    if (this->is_input) {
        this->input_instruction =
            new InputTransferInstruction(
                to_layer, state, environment, compute_stream);
        this->input_copy_instruction =
            new InternalInputTransferInstruction(
                to_layer, state, compute_stream);

        input_instruction->add_dependency(input_copy_instruction);
        input_copy_instruction->add_dependency(input_instruction);
    }

    if (this->is_expected) {
        this->expected_instruction =
            new ExpectedTransferInstruction(
                to_layer, state, environment, compute_stream);
        this->expected_copy_instruction =
            new InternalExpectedTransferInstruction(
                to_layer, state, compute_stream);

        expected_instruction->add_dependency(expected_copy_instruction);
        expected_copy_instruction->add_dependency(expected_instruction);
    }

    // Add noise / clear instruction
    if (to_layer->noise != 0.0)
        instructions.push_back(
            new NoiseInstruction(to_layer, state, compute_stream));
    else if (not this->is_input)
        instructions.push_back(
            new ClearInstruction(to_layer, state, compute_stream));

    // Beform DFS on dendritic tree
    dendrite_DFS(to_layer->dendritic_root);

    // Add state instruction
    this->state_instruction =
        new StateUpdateInstruction(
            to_layer, state, compute_stream);

    // Add output transfer instruction
    if (this->is_output) {
        this->output_copy_instruction =
            new InternalOutputTransferInstruction(
                to_layer, state, compute_stream);
        this->output_instruction =
            new OutputTransferInstruction(
                to_layer, state, environment, io_stream);

        output_copy_instruction->add_dependency(output_instruction);
        output_instruction->add_dependency(output_copy_instruction);
    }
}

ClusterNode::~ClusterNode() {
    for (auto& inst : this->instructions) delete inst;
    if (is_input) {
        delete input_instruction;
        delete input_copy_instruction;
    }
    if (is_expected) {
        delete expected_instruction;
        delete expected_copy_instruction;
    }
    if (is_output) {
        delete output_copy_instruction;
        delete output_instruction;
    }
    delete state_instruction;
}

void ClusterNode::dendrite_DFS(DendriticNode *curr) {
    auto res_man = ResourceManager::get_instance();

    for (auto& child : curr->get_children()) {
        // Create an instruction
        // If internal, recurse first (post-fix DFS)
        if (child->is_leaf()) {
            auto syn_inst = new SynapseInstruction(
                child->conn, state, compute_stream);

            // Check to see if connection is inter-device
            // If so, add an inter-device instruction
            if (state->is_inter_device(child->conn)) {
                // Create transfer instruction
                auto transfer_inst = new DeviceToDeviceTransferFunction(
                        child->conn, state, compute_stream);
                instructions.push_back(transfer_inst);

                // Synapse instruction depends on it
                syn_inst->add_dependency(transfer_inst);
                external_transfer_instructions[child->conn] = transfer_inst;
            }

            // Create the instruction and add it to the synapse instuction list
            instructions.push_back(syn_inst);
            synapse_instructions[child->conn] = syn_inst;
        } else {
            this->dendrite_DFS(child);
            instructions.push_back(
                new DendriticInstruction(
                    curr, child, state, compute_stream));
        }
    }
}

void ClusterNode::activate_input() {
    if (this->is_input) {
        input_instruction->activate();
        input_copy_instruction->activate();
    }
    if (this->is_expected) {
        expected_instruction->activate();
        expected_copy_instruction->activate();
    }
}

void ClusterNode::activate_state() {
    state_instruction->activate();

    if (this->is_output)
        output_copy_instruction->activate();
}

void ClusterNode::activate_output() {
    if (this->is_output)
        output_instruction->activate();
}

void ClusterNode::synchronize_input() {
    if (is_input) input_instruction->synchronize();
    if (is_expected) expected_instruction->synchronize();
}

void ClusterNode::synchronize_output() {
    if (is_output) output_instruction->synchronize();
}

const InstructionList ClusterNode::get_instructions() const {
    return instructions;
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

const std::map<Connection*, Instruction*>
        ClusterNode::get_external_transfer_instructions() {
    return external_transfer_instructions;
}
