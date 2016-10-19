#include <cstdlib>
#include <cstring>
#include "io/buffer.h"
#include "parallel.h"

Buffer::Buffer(int input_start_index, int input_size,
    int output_start_index, int output_size) :
        input_index(input_start_index),
        input_size(input_size),
        output_index(output_start_index),
        output_size(output_size) {
// UNCOMMENT FOR PINNED MEMORY
//#ifdef PARALLEL
//    cudaMallocHost((void**) &this->input, input_size * sizeof(float));
//    cudaMallocHost((void**) &this->output, output_size * sizeof(Output));
//
//    for (int i = 0; i < input_size; ++i) this->input[i] = 0.0;
//#else
    this->input = (float*)calloc(input_size, sizeof(float));
    this->output = (Output*)calloc(output_size, sizeof(Output));
//#endif
}

void Buffer::clear_input() {
    for (int nid = 0 ; nid < this->input_size; ++nid)
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
