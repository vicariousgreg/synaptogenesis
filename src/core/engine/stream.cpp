#include "engine/stream.h"
#include "util/error_manager.h"

Stream::Stream(Layer *layer, State *state, Environment *environment)
        : to_layer(layer),
          state(state),
          environment(environment) {
#ifdef __CUDACC__
    // Create cuda stream
    cudaStreamCreate(&cuda_stream);
    this->external_stream = false;
#endif
    this->init();
}

#ifdef __CUDACC__
Stream::Stream(Layer *layer, State *state, Environment *environment, cudaStream_t cuda_stream)
        : to_layer(layer),
          state(state),
          environment(environment),
          cuda_stream(cuda_stream) {
    this->external_stream = true;
    this->init();
}
#endif

void Stream::init() {
#ifdef __CUDACC__
    cudaEventCreateWithFlags(&finished_event,
        cudaEventDisableTiming);
    cudaEventCreateWithFlags(&input_event,
        cudaEventDisableTiming & cudaEventBlockingSync);
    cudaEventCreateWithFlags(&output_event,
        cudaEventDisableTiming & cudaEventBlockingSync);
    cudaEventCreateWithFlags(&state_event,
        cudaEventDisableTiming & cudaEventBlockingSync);
#endif
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

#ifdef __CUDACC__
    // Add finished event to last synaptic/dendritic connection
    if (instructions.size() > 0)
        instructions[instructions.size()-1]->add_event(finished_event);
#endif
}

Stream::~Stream() {
    for (auto inst : this->instructions) delete inst;
    if (this->input_instruction) delete input_instruction;
    if (this->output_instruction) delete output_instruction;
    delete state_instruction;
#ifdef __CUDACC__
    if (not this->external_stream) cudaStreamDestroy(cuda_stream);
    cudaEventDestroy(finished_event);
#endif
}

void Stream::dendrite_DFS(DendriticNode *curr) {
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

void Stream::set_input_instruction(Instruction *inst) {
    if (input_instruction != NULL)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple input instructions to stream!");
    this->input_instruction = inst;
#ifdef __CUDACC__
    inst->add_event(input_event);
    inst->set_stream(cuda_stream);
#endif
}

void Stream::set_output_instruction(Instruction *inst) {
    if (output_instruction != NULL)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple output instructions to stream!");
    this->output_instruction = inst;
#ifdef __CUDACC__
    inst->add_event(output_event);
    inst->set_stream(state->io_stream);
#endif
}

void Stream::set_state_instruction(Instruction *inst) {
    if (state_instruction != NULL)
        ErrorManager::get_instance()->log_error(
            "Cannot add multiple state instructions to stream!");
    this->state_instruction = inst;
#ifdef __CUDACC__
    inst->add_event(state_event);
    inst->set_stream(cuda_stream);
#endif
}

void Stream::add_instruction(Instruction *inst) {
    this->instructions.push_back(inst);
#ifdef __CUDACC__
    inst->set_stream(this->cuda_stream);
#endif
}

void Stream::activate_input_instruction() {
    if (input_instruction != NULL)
        input_instruction->activate();
#ifdef __CUDACC__
    cudaStreamWaitEvent(cuda_stream, input_event, 0);
#endif
}

void Stream::activate_output_instruction() {
    if (output_instruction != NULL)
        output_instruction->activate();
}

void Stream::activate_state_instruction() {
#ifdef __CUDACC__
    cudaStreamWaitEvent(cuda_stream, output_event, 0);
#endif
    state_instruction->activate();
#ifdef __CUDACC__
    cudaStreamWaitEvent(cuda_stream, state_event, 0);
#endif
}
