#include "engine/stream.h"

Stream::Stream(Layer *layer, State *state) : to_layer(layer), state(state) {
    // Beform DFS on dendritic tree
    dendrite_DFS(to_layer->dendritic_root);

#ifdef PARALLEL
    // Create cuda stream and event
    cudaStreamCreate(&cuda_stream);
    finished_event = new cudaEvent_t;
    cudaEventCreateWithFlags(finished_event, cudaEventDisableTiming);
    instructions[instructions.size()-1]->add_event(finished_event);
#endif
}

Stream::~Stream() {
    for (auto inst : this->instructions) delete inst;
#ifdef PARALLEL
    cudaStreamDestroy(cuda_stream);
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
