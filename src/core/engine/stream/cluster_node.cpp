#include "engine/stream/cluster_node.h"
#include "util/error_manager.h"

ClusterNode::ClusterNode(Layer *layer, State *state, Environment *environment,
        Stream *io_stream, Stream *compute_stream)
        : to_layer(layer),
          is_input(layer->is_input()),
          is_output(layer->is_output()),
          input_event(new Event()),
          input_copy_event(new Event()),
          output_event(new Event()),
          output_copy_event(new Event()),
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
        this->set_input_instruction(new InputTransferInstruction(to_layer, state, environment));
        this->set_input_copy_instruction(new InternalInputTransferInstruction(to_layer, state));
    }

    // Add noise / clear instruction
    if (to_layer->noise != 0.0)
        add_instruction(new NoiseInstruction(to_layer, state));
    else if (not this->is_input)
        add_instruction(new ClearInstruction(to_layer, state));

    // Beform DFS on dendritic tree
    dendrite_DFS(to_layer->dendritic_root);

    // Add state instruction
    this->set_state_instruction(new StateUpdateInstruction(to_layer, state));

    // Add output transfer instruction
    if (this->is_output) {
        this->set_output_instruction(new OutputTransferInstruction(to_layer, state, environment));
        this->set_output_copy_instruction(new InternalOutputTransferInstruction(to_layer, state));
    }
}

ClusterNode::~ClusterNode() {
    for (auto inst : this->instructions) delete inst;
    if (this->input_instruction) delete input_instruction;
    if (this->output_instruction) delete output_instruction;
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
            add_instruction(new SynapseInstruction(child->conn, state));
        } else {
            this->dendrite_DFS(child);
            add_instruction(new DendriticInstruction(curr, child, state));
        }
    }
}

void ClusterNode::set_input_copy_instruction(Instruction *inst) {
    if (input_copy_instruction != nullptr)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple input copy instructions to stream!");
    this->input_copy_instruction = inst;
    inst->set_stream(this->compute_stream);
    inst->add_event(input_copy_event);
}

void ClusterNode::set_input_instruction(Instruction *inst) {
    if (input_instruction != nullptr)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple input instructions to stream!");
    this->input_instruction = inst;
    inst->set_stream(this->compute_stream);
    inst->add_event(input_event);
}

void ClusterNode::set_output_copy_instruction(Instruction *inst) {
    if (output_copy_instruction != nullptr)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple output copy instructions to stream!");
    this->output_copy_instruction = inst;
    inst->set_stream(this->compute_stream);
    inst->add_event(output_copy_event);
}

void ClusterNode::set_output_instruction(Instruction *inst) {
    if (output_instruction != nullptr)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple output instructions to stream!");
    this->output_instruction = inst;
    inst->set_stream(this->io_stream);
    inst->add_event(output_event);
}

void ClusterNode::set_state_instruction(Instruction *inst) {
    if (state_instruction != nullptr)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple state instructions to stream!");
    this->state_instruction = inst;
    inst->set_stream(this->compute_stream);
}

void ClusterNode::add_instruction(Instruction *inst) {
    this->instructions.push_back(inst);
    inst->set_stream(this->compute_stream);
}

void ClusterNode::activate_input_instruction() {
    if (this->is_input) {
        this->wait(input_copy_event);
        input_instruction->activate();
        this->wait(input_event);
        input_copy_instruction->activate();
    }
}

void ClusterNode::activate_output_instruction() {
    if (this->is_output) {
        this->wait(output_copy_event);
        output_instruction->activate();
    }
}

void ClusterNode::activate_state_instruction() {
    state_instruction->activate();
    if (this->is_output) {
        this->wait(output_event);
        output_copy_instruction->activate();
    }
}

void ClusterNode::wait(Event *event) {
    compute_stream->wait(event);
}
