#include "engine/cluster/cluster_node.h"
#include "util/error_manager.h"

ClusterNode::ClusterNode(Layer *layer, State *state, Environment *environment,
        Stream *io_stream, Stream *compute_stream)
        : to_layer(layer),
          is_input(layer->is_input()),
          is_output(layer->is_output()),
          input_event(ResourceManager::get_instance()->create_event()),
          input_copy_event(ResourceManager::get_instance()->create_event()),
          output_event(ResourceManager::get_instance()->create_event()),
          output_copy_event(ResourceManager::get_instance()->create_event()),
          input_instruction(nullptr),
          input_copy_instruction(nullptr),
          output_instruction(nullptr),
          output_copy_instruction(nullptr),
          state_instruction(nullptr),
          io_stream(io_stream),
          compute_stream(compute_stream),
          state(state),
          environment(environment) {
    // Add input transfer instruction
    if (this->is_input) {
        set_input_instruction(
            new InputTransferInstruction(
                to_layer, state, environment, compute_stream));
        set_input_copy_instruction(
            new InternalInputTransferInstruction(
                to_layer, state, compute_stream));
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
    set_state_instruction(
        new StateUpdateInstruction(
            to_layer, state, compute_stream));

    // Add output transfer instruction
    if (this->is_output) {
        set_output_copy_instruction(
            new InternalOutputTransferInstruction(
                to_layer, state, compute_stream));
        set_output_instruction(
            new OutputTransferInstruction(
                to_layer, state, environment, io_stream));
    }
}

ClusterNode::~ClusterNode() {
    for (auto inst : this->instructions) delete inst;
    if (is_input) {
        delete input_instruction;
        delete input_copy_instruction;
    }
    if (is_output) {
        delete output_copy_instruction;
        delete output_instruction;
    }
    delete state_instruction;

    delete input_event;
    delete output_event;
    delete input_copy_event;
    delete output_copy_event;
}

void ClusterNode::dendrite_DFS(DendriticNode *curr) {
    for (auto& child : curr->get_children()) {
        // Create an instruction
        // If internal, recurse first (post-fix DFS)
        if (child->is_leaf()) {
            instructions.push_back(
                new SynapseInstruction(
                    child->conn, state, compute_stream));
        } else {
            this->dendrite_DFS(child);
            instructions.push_back(
                new DendriticInstruction(
                    curr, child, state, compute_stream));
        }
    }
}

void ClusterNode::set_input_instruction(Instruction *inst) {
    if (input_instruction != nullptr)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple input instructions to stream!");
    this->input_instruction = inst;
    inst->add_dependency(input_copy_event);
    inst->add_event(input_event);
}

void ClusterNode::set_input_copy_instruction(Instruction *inst) {
    if (input_copy_instruction != nullptr)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple input copy instructions to stream!");
    this->input_copy_instruction = inst;
    inst->add_dependency(input_event);
    inst->add_event(input_copy_event);
}

void ClusterNode::set_state_instruction(Instruction *inst) {
    if (state_instruction != nullptr)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple state instructions to stream!");
    this->state_instruction = inst;
}

void ClusterNode::set_output_copy_instruction(Instruction *inst) {
    if (output_copy_instruction != nullptr)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple output copy instructions to stream!");
    this->output_copy_instruction = inst;
    inst->add_dependency(output_event);
    inst->add_event(output_copy_event);
}

void ClusterNode::set_output_instruction(Instruction *inst) {
    if (output_instruction != nullptr)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple output instructions to stream!");
    this->output_instruction = inst;
    inst->add_dependency(output_copy_event);
    inst->add_event(output_event);
}

void ClusterNode::activate_input() {
    if (this->is_input) {
        input_instruction->activate();
        input_copy_instruction->activate();
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
    if (is_input) input_event->synchronize();
}

void ClusterNode::synchronize_output() {
    if (is_output) output_event->synchronize();
}
