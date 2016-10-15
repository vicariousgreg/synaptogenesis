#include <cstdlib>
#include <cstring>
#include "io/buffer.h"
#include "parallel.h"

Buffer::Buffer(int num_neurons, int output_size) {
    this->output_size = output_size;
    this->num_neurons = num_neurons;

// UNCOMMENT FOR PINNED MEMORY
//#ifdef PARALLEL
//    cudaMallocHost((void**) &this->input, num_neurons * sizeof(float));
//    cudaMallocHost((void**) &this->output, num_neurons * output_size);
//
//    for (int i = 0; i < num_neurons; ++i) this->input[i] = 0.0;
//#else
    this->input = (float*)calloc(num_neurons, sizeof(float));
    this->output = calloc(num_neurons, output_size);
//#endif
}

void Buffer::send_input_to(State *state) {
#ifdef PARALLEL
    // Copy from GPU to local location
    cudaMemcpy(state->input, this->input,
        this->num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("Failed to copy input from host to device!");
#else
    memcpy(state->input, this->input,
        this->num_neurons * sizeof(float));
#endif
}

void Buffer::retrieve_output_from(State *state) {
#ifdef PARALLEL
    cudaMemcpy(this->output, state->output,
        this->num_neurons * this->output_size,
        cudaMemcpyDeviceToHost);
#else
    memcpy(this->output, state->output,
        this->num_neurons * this->output_size);
#endif
}

void Buffer::clear_input() {
    for (int nid = 0 ; nid < this->num_neurons; ++nid)
        this->input[nid] = 0.0;
}

void Buffer::set_input(int offset, int size, float* source) {
    memcpy(&this->input[offset], source, size * sizeof(*this->input));
}

float* Buffer::get_input() {
    return this->input;
}

void* Buffer::get_output() {
    return this->output;
}
