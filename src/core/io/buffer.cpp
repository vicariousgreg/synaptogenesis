#include <cstdlib>
#include <cstring>

#include "io/buffer.h"
#include "model/model.h"
#include "model/structure.h"

Buffer *build_buffer(DeviceID device_id, Model *model) {
    return build_buffer(device_id,
        model->get_input_layers(),
        model->get_output_layers(),
        model->get_expected_layers());
}

Buffer *build_buffer(DeviceID device_id, LayerList input_layers,
        LayerList output_layers, LayerList expected_layers) {
    return new Buffer(
        input_layers,
        output_layers,
        expected_layers,
        device_id);
}

Buffer::Buffer(LayerList input_layers, LayerList output_layers,
        LayerList expected_layers, DeviceID device_id) : device_id(device_id) {
    this->input_layers = input_layers;
    this->output_layers = output_layers;
    this->expected_layers = expected_layers;
    for (auto layer : input_layers) dirty_map[layer] = false;

    input_size = output_size = expected_size = 0;
    for (auto layer : input_layers) input_size += layer->size;
    for (auto layer : output_layers) output_size += layer->size;
    for (auto layer : expected_layers) expected_size += layer->size;

    // Create pointers
    // Use pinned pointers for host
    if (ResourceManager::get_instance()->is_host(device_id)) {
        if (input_size > 0)
            input = Pointer<float>::pinned_pointer(input_size, 0.0);
        if (output_size > 0)
            output = Pointer<Output>::pinned_pointer(output_size);
        if (expected_size > 0)
            expected = Pointer<Output>::pinned_pointer(expected_size);
    } else {
        if (input_size > 0)
            input = Pointer<float>(input_size, 0.0);
        if (output_size > 0)
            output = Pointer<Output>(output_size);
        if (expected_size > 0)
            expected = Pointer<Output>(expected_size);
    }

    // Set up maps
    int input_index = 0;
    int output_index = 0;
    int expected_index = 0;
    for (auto& layer : input_layers) {
        input_map[layer] = input_index;
        input_index += layer->size;
    }
    for (auto& layer : output_layers) {
        output_map[layer] = output_index;
        output_index += layer->size;
    }
    for (auto& layer : expected_layers) {
        expected_map[layer] = expected_index;
        expected_index += layer->size;
    }
}

Buffer::~Buffer() {
    this->input.free();
    this->output.free();
    this->expected.free();
}

void Buffer::set_input(Layer* layer, Pointer<float> source) {
    source.copy_to(this->get_input(layer));
    dirty_map[layer] = true;
}

void Buffer::set_output(Layer* layer, Pointer<Output> source) {
    source.copy_to(this->get_output(layer));
}

void Buffer::set_expected(Layer* layer, Pointer<Output> source) {
    source.copy_to(this->get_expected(layer));
}

Pointer<float> Buffer::get_input(Layer *layer) {
    return input.slice(input_map[layer], layer->size);
}

Pointer<Output> Buffer::get_output(Layer *layer) {
    return output.slice(output_map[layer], layer->size);
}

Pointer<Output> Buffer::get_expected(Layer *layer) {
    return expected.slice(expected_map[layer], layer->size);
}

std::vector<BasePointer*> Buffer::get_pointers() {
    std::vector<BasePointer*> pointers = {
        &input, &output, &expected
    };
    return pointers;
}
