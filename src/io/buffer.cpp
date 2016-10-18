#include <cstdlib>
#include <cstring>
#include "io/buffer.h"
#include "parallel.h"

Buffer::Buffer(int num_neurons) {
    this->num_neurons = num_neurons;

// UNCOMMENT FOR PINNED MEMORY
//#ifdef PARALLEL
//    cudaMallocHost((void**) &this->input, num_neurons * sizeof(float));
//    cudaMallocHost((void**) &this->output, num_neurons * sizeof(Output));
//
//    for (int i = 0; i < num_neurons; ++i) this->input[i] = 0.0;
//#else
    this->input = (float*)calloc(num_neurons, sizeof(float));
    this->output = (Output*)calloc(num_neurons, sizeof(Output));
//#endif
}

void Buffer::clear_input() {
    for (int nid = 0 ; nid < this->num_neurons; ++nid)
        this->input[nid] = 0.0;
}

void Buffer::set_input(int offset, int size, float* source) {
    memcpy(&this->input[offset], source, size * sizeof(float));
}

void Buffer::set_output(int offset, int size, Output* source) {
    memcpy(&this->output[offset], source, size * sizeof(Output));
}

float* Buffer::get_input() {
    return this->input;
}

Output* Buffer::get_output() {
    return this->output;
}
