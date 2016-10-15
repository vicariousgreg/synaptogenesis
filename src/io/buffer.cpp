#include <cstdlib>
#include <cstring>
#include "io/buffer.h"
#include "parallel.h"

Buffer::Buffer(int num_neurons, int output_size) {
    this->output_size = output_size;
    this->num_neurons = num_neurons;
    this->input = (float*)calloc(num_neurons, sizeof(*input));
    this->output = calloc(num_neurons, output_size);
}

void Buffer::clear_input() {
    for (int nid = 0 ; nid < this->num_neurons; ++nid)
        this->input[nid] = 0.0;
}

void Buffer::set_input(int offset, int size, float* source) {
    memcpy(&this->input[offset], source, size * sizeof(*this->input));
}

void Buffer::set_output(int offset, int size, void* source) {
#ifdef PARALLEL
    cudaMemcpy(this->output + (offset * this->output_size),
        source, size * this->output_size,
        cudaMemcpyDeviceToHost);
#else
    memcpy(this->output + (offset * this->output_size),
        source, size * this->output_size);
#endif
}

float* Buffer::get_input() {
    return this->input;
}

void* Buffer::get_output() {
    return this->output;
}
