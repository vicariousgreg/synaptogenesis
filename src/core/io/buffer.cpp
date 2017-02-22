#include <cstdlib>
#include <cstring>

#include "io/buffer.h"
#include "model/structure.h"
#include "util/parallel.h"

Buffer::Buffer(Structure* structure, OutputType output_type) :
        output_type(output_type) {
    int input_output_size = structure->get_num_neurons(INPUT_OUTPUT);
    int input_size = input_output_size + structure->get_num_neurons(INPUT);
    int output_size = input_output_size + structure->get_num_neurons(OUTPUT);

#ifdef PARALLEL
    // Allocate pinned memory
    if (input_size > 0)
        cudaMallocHost((void**) &this->input, input_size * sizeof(float));
    else this->input = NULL;
    if (output_size > 0)
        cudaMallocHost((void**) &this->output, output_size * sizeof(Output));
    else this->output = NULL;

    for (int i = 0; i < input_size; ++i) this->input[i] = 0.0;
#else
    // Allocate unpinned memory
    if (input_size > 0)
        this->input = (float*)calloc(input_size, sizeof(float));
    else this->input = NULL;

    if (output_size > 0)
        this->output = (Output*)calloc(output_size, sizeof(Output));
    else this->output = NULL;
#endif

    // Set up maps
    int input_index = 0;
    int output_index = 0;
    for (auto& layer : structure->get_layers(INPUT)) {
        input_map[layer] = input + input_index;
        input_index += layer->size;
    }
    for (auto& layer : structure->get_layers(INPUT_OUTPUT)) {
        input_map[layer] = input + input_index;
        input_index += layer->size;
        output_map[layer] = output + output_index;
        output_index += layer->size;
    }
    for (auto& layer : structure->get_layers(OUTPUT)) {
        output_map[layer] = output + output_index;
        output_index += layer->size;
    }
}

Buffer::~Buffer() {
#ifdef PARALLEL
    // Free pinned memory
    if (input_size > 0) cudaFreeHost(this->input);
    if (output_size > 0) cudaFreeHost(this->output);
#else
    // Free non-pinned memory
    if (input_size > 0) free(this->input);
    if (output_size > 0) free(this->output);
#endif
}

void Buffer::set_input(Layer* layer, float* source) {
    memcpy(this->get_input(layer), source, layer->size * sizeof(float));
}

void Buffer::set_output(Layer* layer, Output* source) {
    memcpy(this->get_output(layer), source, layer->size * sizeof(Output));
}
