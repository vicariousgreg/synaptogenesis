#include "engine/stream/cluster_node.h"
#include "util/error_manager.h"

ClusterNode::ClusterNode(Layer *layer, State *state, Environment *environment,
        Stream *io_stream, Stream *compute_stream)
        : to_layer(layer),
          io_stream(io_stream),
          state(state),
          environment(environment) {
    if (compute_stream == nullptr) {
        this->compute_stream = new Stream();
        this->external_stream = false;
    } else {
        this->compute_stream = compute_stream;
        this->external_stream = true;
    }

    this->finished_event = new Event();
    this->input_event = new Event();
    this->output_event = new Event();
    this->state_event = new Event();

    input_instruction = NULL;
    output_instruction = NULL;
    state_instruction = NULL;

    // Add input transfer instruction
    if (to_layer->get_input_module() != NULL)
        this->set_input_instruction(new InputTransferInstruction(to_layer, state, environment));
    // Add output transfer instruction
    if (to_layer->get_output_modules().size() > 0)
        this->set_output_instruction(new OutputTransferInstruction(to_layer, state, environment));
    // Add state instruction
    this->set_state_instruction(new StateUpdateInstruction(to_layer, state));
    // Add noise / clear instruction
    if (to_layer->noise != 0.0)
        add_instruction(new NoiseInstruction(to_layer, state));
    else if (to_layer->get_input_module() == NULL)
        add_instruction(new ClearInstruction(to_layer, state));

    // Beform DFS on dendritic tree
    dendrite_DFS(to_layer->dendritic_root);

    // Add finished event to last synaptic/dendritic connection
    if (instructions.size() > 0)
        instructions[instructions.size()-1]->add_event(finished_event);
}

ClusterNode::~ClusterNode() {
    for (auto inst : this->instructions) delete inst;
    if (this->input_instruction) delete input_instruction;
    if (this->output_instruction) delete output_instruction;
    delete state_instruction;
    if (not this->external_stream) delete compute_stream;

    delete finished_event;
    delete input_event;
    delete output_event;
    delete state_event;
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

void ClusterNode::set_input_instruction(Instruction *inst) {
    if (input_instruction != NULL)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple input instructions to stream!");
    this->input_instruction = inst;
    inst->set_stream(this->compute_stream);
    inst->add_event(input_event);
}

void ClusterNode::set_output_instruction(Instruction *inst) {
    if (output_instruction != NULL)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple output instructions to stream!");
    this->output_instruction = inst;
    inst->set_stream(this->io_stream);
    inst->add_event(output_event);
}

void ClusterNode::set_state_instruction(Instruction *inst) {
    if (state_instruction != NULL)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple state instructions to stream!");
    this->state_instruction = inst;
    inst->set_stream(this->compute_stream);
    inst->add_event(state_event);
}

void ClusterNode::add_instruction(Instruction *inst) {
    this->instructions.push_back(inst);
    inst->set_stream(this->compute_stream);
}

void ClusterNode::activate_input_instruction() {
    if (input_instruction != NULL)
        input_instruction->activate();
    compute_stream->wait(input_event);
}

void ClusterNode::activate_output_instruction() {
    if (output_instruction != NULL)
        output_instruction->activate();
}

void ClusterNode::activate_state_instruction() {
    compute_stream->wait(output_event);
    state_instruction->activate();
    compute_stream->wait(state_event);
}
