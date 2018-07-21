#include <cstdlib>
#include <cstring>

#include "io/buffer.h"
#include "network/network.h"
#include "network/structure.h"
#include "util/logger.h"

Buffer *build_buffer(DeviceID device_id,
        LayerList input_layers, LayerList output_layers,
        LayerKeyMap input_keys, LayerKeyMap output_keys) {
    // Ensure that input/output layers have a key (assume default)
    for (auto layer : input_layers)
        if (input_keys[layer].size() == 0)
            input_keys[layer].insert("input");

    for (auto layer : output_layers)
        if (output_keys[layer].size() == 0)
            output_keys[layer].insert("output");

    return new Buffer(
        device_id,
        input_layers,
        output_layers,
        input_keys,
        output_keys);
}

Buffer::Buffer(DeviceID device_id,
        LayerList input_layers, LayerList output_layers,
        LayerKeyMap input_keys, LayerKeyMap output_keys)
            : device_id(device_id) {
    bool is_host = ResourceManager::get_instance()->is_host(device_id);

    for (auto layer : input_layers) {
        if (input_keys[layer].count("input")) {
            if (is_host) {
                auto ptr = Pointer<float>::pinned_pointer(layer->size, 0.0);
                input[layer] = new Pointer<float>(ptr, true);
            } else {
                input[layer] = new Pointer<float>(layer->size, 0.0);
            }
            input_dirty_map[layer] = false;
        }
    }

    for (auto layer_pair : input_keys) {
        auto layer = layer_pair.first;

        for (auto key : layer_pair.second) {
            if (key == "input") continue;

            if (is_host) {
                auto ptr = Pointer<float>::pinned_pointer(layer->size, 0.0);
                input_auxiliary[layer][key] = new Pointer<float>(ptr, true);
            } else {
                input_auxiliary[layer][key] =
                    new Pointer<float>(layer->size, 0.0);
            }
            auxiliary_dirty_map[layer][key] = false;
        }
    }

    for (auto layer : output_layers) {
        if (output_keys[layer].count("output")) {
            if (is_host) {
                auto ptr = Pointer<Output>::pinned_pointer(layer->size);
                output[layer] = new Pointer<Output>(ptr, true);
            } else {
                output[layer] = new Pointer<Output>(layer->size);
            }
        }
    }

    for (auto layer_pair : output_keys) {
        auto layer = layer_pair.first;

        for (auto key : layer_pair.second) {
            if (key == "output") continue;

            if (is_host) {
                auto ptr = Pointer<Output>::pinned_pointer(layer->size);
                output_auxiliary[layer][key] = new Pointer<Output>(ptr, true);
            } else {
                output_auxiliary[layer][key] =
                    new Pointer<Output>(layer->size);
            }
        }
    }
}

Buffer::~Buffer() {
    for (auto layer_pair : input) {
        layer_pair.second->free();
        delete layer_pair.second;
    }

    for (auto layer_pair : output) {
        layer_pair.second->free();
        delete layer_pair.second;
    }

    for (auto layer_pair : input_auxiliary) {
        for (auto key_pair : layer_pair.second) {
            key_pair.second->free();
            delete key_pair.second;
        }
    }

    for (auto layer_pair : output_auxiliary) {
        for (auto key_pair : layer_pair.second) {
            key_pair.second->free();
            delete key_pair.second;
        }
    }
}

std::vector<BasePointer*> Buffer::get_pointers() {
    std::vector<BasePointer*> pointers;

    for (auto layer_pair : input)
        pointers.push_back(layer_pair.second);
    for (auto layer_pair : output)
        pointers.push_back(layer_pair.second);

    for (auto layer_pair : input_auxiliary)
        for (auto key_pair : layer_pair.second)
            pointers.push_back(key_pair.second);
    for (auto layer_pair : output_auxiliary)
        for (auto key_pair : layer_pair.second)
            pointers.push_back(key_pair.second);

    return pointers;
}

void Buffer::set_input(Layer* layer, Pointer<float> source) {
    source.copy_to(this->get_input(layer));
}

void Buffer::set_output(Layer* layer, Pointer<Output> source) {
    source.copy_to(this->get_output(layer));
}

Pointer<float> Buffer::get_input(Layer *layer) {
    try {
        // Assume that the input is dirty if pointer is retrieved
        input_dirty_map[layer] = true;
        return *input.at(layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Attempted to retrieve input from Buffer for "
            "unrepresented layer: " + layer->str());
    }
}

Pointer<Output> Buffer::get_output(Layer *layer) {
    try {
        return *output.at(layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Attempted to retrieve output from Buffer for "
            "unrepresented layer: " + layer->str());
    }
}

BasePointer* Buffer::get_input_auxiliary(Layer *layer, std::string key) {
    try {
        // Assume that the input is dirty if pointer is retrieved
        if (key == "input") {
            input_dirty_map[layer] = true;
            return input.at(layer);
        } else {
            auxiliary_dirty_map[layer][key] = true;
            return input_auxiliary[layer][key];
        }
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve auxiliary input \"" + key +
            "\" from Buffer for layer: " + layer->str());
    }
}

BasePointer* Buffer::get_output_auxiliary(Layer *layer, std::string key) {
    try {
        if (key == "output") return output.at(layer);
        else return output_auxiliary[layer][key];
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve auxiliary output \"" + key +
            "\" from Buffer for layer: " + layer->str());
    }
}

bool Buffer::get_input_dirty(Layer *layer) const {
    try {
        return input_dirty_map.at(layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Attempted to retrieve dirty flag from Buffer for "
            "unrepresented layer: " + layer->str());
    }
}
bool Buffer::set_input_dirty(Layer *layer, bool dirty) {
    input_dirty_map[layer] = dirty;
}

bool Buffer::get_auxiliary_dirty(Layer *layer, std::string key) const {
    try {
        return auxiliary_dirty_map.at(layer).at(key);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Attempted to retrieve dirty flag from Buffer for "
            "unrepresented layer: " + layer->str());
    }
}
bool Buffer::set_auxiliary_dirty(Layer *layer, std::string key,
        bool dirty) {
    auxiliary_dirty_map[layer][key] = dirty;
}
