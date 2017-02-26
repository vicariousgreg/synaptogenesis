#include <cstdlib>
#include <cstring>

#include "io/buffer.h"
#include "model/model.h"
#include "model/structure.h"

Buffer::Buffer(Model *model) {
    LayerList input_layers, output_layers;

    // Extract layers from model
    for (auto layer : model->get_layers()) {
        if (layer->is_input()) input_layers.push_back(layer);
        if (layer->is_output()) output_layers.push_back(layer);
    }

    this->init(input_layers, output_layers);
}

Buffer::Buffer(Structure *structure) {
    LayerList input_layers, output_layers;

    // Extract layers from structure
    for (auto layer : structure->get_layers()) {
        if (layer->is_input()) input_layers.push_back(layer);
        if (layer->is_output()) output_layers.push_back(layer);
    }

    this->init(input_layers, output_layers);
}

Buffer::Buffer(LayerList layers) {
    this->init(layers, layers);
}

Buffer::Buffer(LayerList input_layers, LayerList output_layers) {
    this->init(input_layers, output_layers);
}

Buffer::~Buffer() {
    this->input.free();
    this->output.free();
}

void Buffer::init(LayerList input_layers, LayerList output_layers) {
    input_size = output_size = 0;
    for (auto layer : input_layers) input_size += layer->size;
    for (auto layer : output_layers) output_size += layer->size;

    // Allocate buffer memory
    if (input_size > 0) input = Pointer<float>::pinned_pointer(input_size, 0.0);
    if (output_size > 0) output = Pointer<Output>::pinned_pointer(output_size);

    // Set up maps
    int input_index = 0;
    int output_index = 0;
    for (auto& layer : input_layers) {
        input_map[layer] = input.slice(input_index, layer->size);
        input_index += layer->size;
    }
    for (auto& layer : output_layers) {
        output_map[layer] = output.slice(output_index, layer->size);
        output_index += layer->size;
    }
}

void Buffer::set_input(Layer* layer, Pointer<float> source) {
    source.copy_to(this->get_input(layer), false);
}

void Buffer::set_output(Layer* layer, Pointer<Output> source) {
    source.copy_to(this->get_output(layer), false);
}
