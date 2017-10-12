#include <cstdlib>
#include <cstring>

#include "io/buffer.h"
#include "network/network.h"
#include "network/structure.h"
#include "util/error_manager.h"

Buffer *build_buffer(DeviceID device_id, LayerList input_layers,
        LayerList output_layers, LayerList expected_layers) {
    return new Buffer(
        input_layers,
        output_layers,
        expected_layers,
        device_id);
}

Buffer::Buffer(LayerList input_layers, LayerList output_layers, LayerList expected_layers,
        DeviceID device_id) : device_id(device_id) {
    input_size = output_size = expected_size = 0;
    for (auto layer : input_layers) {
        input_size += layer->size;
        dirty_map[layer] = false;
    }
    for (auto layer : expected_layers) {
        expected_size += layer->size;
        dirty_map[layer] = false;
    }
    for (auto layer : output_layers)
        output_size += layer->size;


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

std::vector<BasePointer*> Buffer::get_pointers() {
    std::vector<BasePointer*> pointers = {
        &input, &output, &expected
    };
    return pointers;
}

void Buffer::set_input(Layer* layer, Pointer<float> source) {
    source.copy_to(this->get_input(layer));
}

void Buffer::set_output(Layer* layer, Pointer<Output> source) {
    source.copy_to(this->get_output(layer));
}

void Buffer::set_expected(Layer* layer, Pointer<Output> source) {
    source.copy_to(this->get_expected(layer));
}

Pointer<float> Buffer::get_input(Layer *layer) {
    try {
        // Assume that the input is dirty if pointer is retrieved
        dirty_map[layer] = true;
        return input.slice(input_map.at(layer), layer->size);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Attempted to retrieve input from Buffer for "
            "unrepresented layer: " + layer->str());
    }
}

Pointer<Output> Buffer::get_output(Layer *layer) {
    try {
        return output.slice(output_map.at(layer), layer->size);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Attempted to retrieve output from Buffer for "
            "unrepresented layer: " + layer->str());
    }
}

Pointer<Output> Buffer::get_expected(Layer *layer) {
    try {
        return expected.slice(expected_map.at(layer), layer->size);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Attempted to retrieve expected data from Buffer for "
            "unrepresented layer: " + layer->str());
    }
}

bool Buffer::get_dirty(Layer *layer) const {
    try {
        return dirty_map.at(layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Attempted to retrieve dirty flag from Buffer for "
            "unrepresented layer: " + layer->str());
    }
}
bool Buffer::set_dirty(Layer *layer, bool dirty) {
    dirty_map[layer] = dirty;
}
