#include <cstdlib>
#include <cstring>

#include "io/buffer.h"
#include "model/model.h"
#include "model/structure.h"

Buffer *build_buffer(DeviceID device_id, Model *model) {
    return build_buffer(device_id,
        model->get_input_layers(), model->get_output_layers());
}

Buffer *build_buffer(DeviceID device_id,
        LayerList input_layers, LayerList output_layers) {
    if (ResourceManager::get_instance()->is_host(device_id))
        return new HostBuffer(input_layers, output_layers);
    else
        return new DeviceBuffer(input_layers, output_layers, device_id);
}

Buffer::Buffer(LayerList input_layers, LayerList output_layers) {
    this->input_layers = input_layers;
    this->output_layers = output_layers;
    for (auto layer : input_layers) dirty_map[layer] = false;

    input_size = output_size = 0;
    for (auto layer : input_layers) input_size += layer->size;
    for (auto layer : output_layers) output_size += layer->size;
}

void Buffer::init() {
    // Set up maps
    int input_index = 0;
    int output_index = 0;
    for (auto& layer : input_layers) {
        input_map[layer] = input_index;
        input_index += layer->size;
    }
    for (auto& layer : output_layers) {
        output_map[layer] = output_index;
        output_index += layer->size;
    }
}

Buffer::~Buffer() {
    this->input.free();
    this->output.free();
}

void Buffer::set_input(Layer* layer, Pointer<float> source) {
    source.copy_to(this->get_input(layer));
    dirty_map[layer] = true;
}

void Buffer::set_output(Layer* layer, Pointer<Output> source) {
    source.copy_to(this->get_output(layer));
}

Pointer<float> Buffer::get_input(Layer *layer) {
    return input.slice(input_map[layer], layer->size);
}

Pointer<Output> Buffer::get_output(Layer *layer) {
    return output.slice(output_map[layer], layer->size);
}

void HostBuffer::init() {
    // Allocate buffer memory
    if (input_size > 0) input = Pointer<float>::pinned_pointer(input_size, 0.0);
    if (output_size > 0) output = Pointer<Output>::pinned_pointer(output_size);
    Buffer::init();
}

void DeviceBuffer::init() {
    // Allocate buffer memory
    if (input_size > 0) {
        input = Pointer<float>(input_size, 0.0);
        input.schedule_transfer(device_id);
    }
    if (output_size > 0) {
        output = Pointer<Output>(output_size);
        output.schedule_transfer(device_id);
    }
    Buffer::init();
}
