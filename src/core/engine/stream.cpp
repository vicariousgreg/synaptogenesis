#include "engine/stream.h"

Stream::Stream(Layer *layer, State *state) : to_layer(layer), state(state) {
    // Add initialization instruction if necessary
    if (layer->noise != 0.0)
        add_instruction(new RandomizeInstruction(layer, state));
    else if (layer->get_input_module() == NULL)
        add_instruction(new ClearInstruction(layer, state));

    // Beform DFS on dendritic tree
    dendrite_DFS(to_layer->dendritic_root);

#ifdef PARALLEL
    // Create cuda stream and event
    cudaStreamCreate(&cuda_stream);
    finished_event = new cudaEvent_t;
    cudaEventCreateWithFlags(finished_event, cudaEventDisableTiming);
    if (instructions.size() > 0)
        instructions[instructions.size()-1]->add_event(finished_event);
    this->external_stream = false;
#endif
}

#ifdef PARALLEL
Stream::Stream(Layer *layer, State *state, cudaStream_t cuda_stream) :
        to_layer(layer),
        state(state),
        cuda_stream(cuda_stream) {
    // Beform DFS on dendritic tree
    dendrite_DFS(to_layer->dendritic_root);

    // Create cuda event
    finished_event = new cudaEvent_t;
    cudaEventCreateWithFlags(finished_event, cudaEventDisableTiming);
    if (instructions.size() > 0)
        instructions[instructions.size()-1]->add_event(finished_event);
    this->external_stream = true;
}
#endif

Stream::~Stream() {
    for (auto inst : this->instructions) delete inst;
#ifdef PARALLEL
    if (not this->external_stream) cudaStreamDestroy(cuda_stream);
    cudaEventDestroy(*finished_event);
    delete finished_event;
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

void Stream::add_instruction(Instruction *inst) {
    this->instructions.push_back(inst);
#ifdef PARALLEL
    inst->set_stream(&this->cuda_stream);
#endif
}
