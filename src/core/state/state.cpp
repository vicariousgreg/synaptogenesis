#include <cstring>
#include <fstream>
#include <sys/stat.h>

#include "state/state.h"
#include "state/weight_matrix.h"
#include "network/network.h"
#include "io/buffer.h"
#include "util/tools.h"

static std::map<Layer*, DeviceID> distribute_layers(
        const LayerList& layers, std::vector<DeviceID> devices) {
    std::map<Layer*, DeviceID> layer_devices;

    // Distribute layers
    // Count up weights
    std::map<Layer*, int> num_weights;
    for (auto layer : layers) {
        num_weights[layer] = 0;
        for (auto& conn : layer->get_input_connections())
            num_weights[layer] += conn->get_compute_weights();
    }

    // Keep track of weight distribution to devices
    std::vector<int> device_weights;
    for (auto device : devices)
        device_weights.push_back(0);

    // Give the next biggest layer to the device with the least weight
    //   until no layers are left to distribute
    for (int i = 0; num_weights.size() > 0; ++i) {
        // Typically display device is 0, so start at the other end
        // This helps avoid burdening the display device in some situations
        int next_device = devices[devices.size()-1];
        for (int i = 0 ; i < device_weights.size(); ++i)
            if (device_weights[i] < device_weights[next_device])
                next_device = i;

        Layer *biggest;
        int size = -1;
        for (auto pair : num_weights) {
            if (pair.second > size) {
                size = pair.second;
                biggest = pair.first;
            }
        }
        num_weights.erase(biggest);
        layer_devices[biggest] = next_device;
        device_weights[next_device] += size;
    }

    return layer_devices;
}

State::State(Network *network) : network(network), on_host(true) {
    // Determine number of non-host devices
    // Subtract one if cuda is enabled to avoid distribution to the host
    this->active_devices = ResourceManager::get_instance()->get_active_devices();

    // Add pointer vectors for each device
    for (auto device : active_devices) {
        network_pointers[device] = std::vector<BasePointer*>();
        buffer_pointers[device] = std::vector<BasePointer*>();
    }

    // Distribute layers
    this->layer_devices = distribute_layers(network->get_layers(), active_devices);

    // Validate neural model strings
    for (auto layer : network->get_layers())
        if (Attributes::get_neural_models().count(layer->neural_model) == 0)
            LOG_ERROR(
                "Unrecognized neural model \"" + layer->neural_model +
                "\" in " + layer->str());

    // Create attributes and weight matrices
    for (auto device_id : active_devices) {
        attributes[device_id] = std::map<std::string, Attributes*>();
        for (auto neural_model : Attributes::get_neural_models()) {
            LayerList layers;
            for (auto layer : network->get_layers(neural_model))
                if (layer_devices[layer] == device_id)
                    layers.push_back(layer);

            if (layers.size() == 0) {
                attributes[device_id][neural_model] = nullptr;
            } else {
                auto att = Attributes::build_attributes(layers, neural_model, device_id);
                attributes[device_id][neural_model] = att;

                // Retrieve pointers
                for (auto ptr : att->get_pointers())
                    network_pointers[device_id].push_back(ptr);
                for (auto pair : att->get_pointer_map())
                    pointer_map[pair.first] = pair.second;

                // Set up weight matrices
                for (auto& layer : layers) {
                    for (auto& conn : layer->get_input_connections()) {
                        WeightMatrix* matrix = new WeightMatrix(conn,
                            att->get_matrix_depth(conn), device_id);
                        this->weight_matrices[conn] = matrix;
                        att->process_weight_matrix(matrix);
                        auto ptr = matrix->get_pointer();
                        network_pointers[device_id].push_back(ptr);
                        pointer_map[PointerKey(
                            conn->id, "matrix", ptr->get_bytes(), 0)] = ptr;
                    }
                }
            }
        }
    }

    // Create buffers
    for (auto device_id : active_devices) {
        // Set up internal buffer
        LayerList device_layers;

        // Extract layers assigned to this device
        for (auto layer : network->get_layers())
            if (layer_devices[layer] == device_id)
                device_layers.push_back(layer);

        // Set up input/expected buffer for device layers
        // No need for output, which is streamed straight off device
        auto buffer =
            build_buffer(device_id, device_layers, LayerList(), device_layers);
        internal_buffers[device_id] = buffer;

        // Retrieve pointers
        for (auto ptr : buffer->get_pointers())
            buffer_pointers[device_id].push_back(ptr);

        // Set up inter-device buffers (one per word index)
        std::map<int, LayerList> inter_device_layers;

        // Identify any inter-device connections and make room for them
        for (auto layer : network->get_layers())
            if (layer_devices[layer] == device_id)
                for (auto conn : layer->get_input_connections())
                    if (is_inter_device(conn)) {
                        int word_index =
                            get_word_index(conn->delay,
                                Attributes::get_output_type(conn->from_layer));
                        inter_device_layers[word_index].push_back(conn->from_layer);
                    }

        // Create a map of word indices to buffers
        std::map<int, Buffer*> buffer_map;
        for (auto pair : inter_device_layers) {
            auto buffer = build_buffer(device_id, LayerList(), pair.second, LayerList());
            buffer_map[pair.first] = buffer;

            // Retrieve pointers
            for (auto ptr : buffer->get_pointers())
                buffer_pointers[device_id].push_back(ptr);
        }

        // Set the inter device buffer
        inter_device_buffers[device_id] = buffer_map;
    }
}

State::~State() {
    for (auto pair : attributes)
        for (auto neural_model : Attributes::get_neural_models())
            if (pair.second[neural_model] != nullptr)
                delete pair.second[neural_model];
    for (auto matrix : weight_matrices) delete matrix.second;
    for (auto buffer : internal_buffers) delete buffer.second;
    for (auto map : inter_device_buffers)
        for (auto pair : map.second)
            delete pair.second;
    for (auto ptr : data_block_pointers) {
        ptr->free();
        delete ptr;
    }
}

void State::transfer_to_device() {
#ifdef __CUDACC__
    if (not on_host) return;

    auto res_man = ResourceManager::get_instance();
    DeviceID host_id = res_man->get_host_id();

    // Accumulate new data block pointers
    std::set<BasePointer*> new_data_block_pointers;

    // Transfer network pointers
    for (auto pair : network_pointers) {
        auto device_id = pair.first;
        if (device_id != host_id and pair.second.size() > 0)
            new_data_block_pointers.insert(
                res_man->transfer(device_id, pair.second));
    }

    // Transfer buffer pointers
    for (auto pair : buffer_pointers) {
        auto device_id = pair.first;
        if (device_id != host_id and pair.second.size() > 0)
            new_data_block_pointers.insert(
                res_man->transfer(device_id, pair.second));
    }

    // Transfer attributes
    for (auto n : Attributes::get_neural_models())
        for (auto pair : attributes)
            if (pair.second[n] != nullptr)
                pair.second[n]->transfer_to_device();

    // Free old data block pointers and replace with new ones
    for (auto ptr : data_block_pointers) ptr->free();
    data_block_pointers = new_data_block_pointers;

    on_host = false;
#endif
}

void State::transfer_to_host() {
#ifdef __CUDACC__
    if (on_host) return;

    // Accumulate new data block pointers
    std::set<BasePointer*> new_data_block_pointers;

    auto res_man = ResourceManager::get_instance();
    DeviceID host_id = res_man->get_host_id();

    // Transfer network pointers
    for (auto pair : network_pointers) {
        auto device_id = pair.first;
        if (device_id != host_id and pair.second.size() > 0)
            new_data_block_pointers.insert(
                res_man->transfer(host_id, pair.second));
    }

    // Transfer buffer pointers
    for (auto pair : buffer_pointers) {
        auto device_id = pair.first;
        if (device_id != host_id and pair.second.size() > 0)
            new_data_block_pointers.insert(
                res_man->transfer(host_id, pair.second));
    }

    // Free old data block pointers and replace with new ones
    for (auto ptr : data_block_pointers) ptr->free();
    data_block_pointers = new_data_block_pointers;

    on_host = true;
#endif
}

void State::copy_to(State* other) {
    // Transfer states to the host if necessary
    if (not this->on_host) this->transfer_to_host();
    if (not other->on_host) other->transfer_to_host();

    // Copy over any matching pointers
    for (auto pair : this->pointer_map) {
        if (other->pointer_map.count(pair.first) > 0) {
            auto this_ptr = this->pointer_map.at(pair.first);
            auto other_ptr = other->pointer_map.at(pair.first);
            if (this_ptr->get_bytes() == other_ptr->get_bytes())
                this_ptr->copy_to(other_ptr);
        }
    }
}

size_t State::get_network_bytes() const {
    size_t size = 0;
    for (auto pair : network_pointers)
        for (auto ptr : pair.second)
            size += ptr->get_bytes();
    return size;
}

size_t State::get_buffer_bytes() const {
    size_t size = 0;
    for (auto pair : buffer_pointers)
        for (auto ptr : pair.second)
            size += ptr->get_bytes();
    return size;
}

bool State::exists(std::string file_name) {
    std::ifstream f(("./states/" + file_name).c_str());
    return f.good();
}

void State::save(std::string file_name, bool verbose) {
    // Transfer to host
    this->transfer_to_host();

    // Open file stream
    std::string path = "./states/" + file_name;
    std::ofstream output_file(path, std::ofstream::binary);
    if (verbose)
        printf("Saving network state to %s ...\n", path.c_str());

    for (auto pair : pointer_map) {
        size_t bytes = pair.first.bytes;
        size_t offset = pair.first.offset;
        const char* data = (const char*)pair.second->get(offset);

        if (not output_file.write((const char*)&pair.first, sizeof(PointerKey)))
            LOG_ERROR(
                "Error writing state to file!");
        if (not output_file.write(data, bytes))
            LOG_ERROR(
                "Error writing state to file!");
    }

    // Close file stream
    output_file.close();
}

void State::load(std::string file_name, bool verbose) {
    // Transfer to host
    this->transfer_to_host();

    // Open file stream
    std::string path = "./states/" + file_name;
    std::ifstream input_file(path, std::ifstream::binary);
    if (verbose)
        printf("Loading network state from %s ...\n", path.c_str());

    // Determine file length
    input_file.seekg (0, input_file.end);
    auto length = input_file.tellg();
    input_file.seekg (0, input_file.beg);

    unsigned long read = 0;

    while (read < length) {
        PointerKey key(0,0,0,0);
        if (not input_file.read((char*)&key, sizeof(PointerKey)))
            LOG_ERROR(
                "Error reading pointer key from file!");

        char* ptr;
        try {
            ptr = (char*) pointer_map.at(key)->get(key.offset);
        } catch (...) {
            LOG_WARNING(
                "Error retrieving pointer -- continuing...");
            continue;
        }

        if (not input_file.read(ptr, key.bytes))
            LOG_ERROR(
                "Error reading data from file!");

        read += sizeof(PointerKey) + key.bytes;
    }

    // Close file stream
    input_file.close();
}

bool State::check_compatibility(Structure *structure) {
    // Check relevant attributes for compatibility
    for (auto n : Attributes::get_neural_models())
        for (auto pair : attributes)
            if (structure->contains(n) and pair.second[n] and not
                    pair.second[n]->check_compatibility(
                        structure->cluster_type))
                return false;
    return true;
}

/* Zoo of Getters */
Pointer<float> State::get_input(Layer *layer, int register_index) const {
    try {
        return attributes.at(layer_devices.at(layer)).at(layer->neural_model)
            ->get_input(layer->id, register_index);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get input data in State for "
            "layer: " + layer->str()
            + ", index: " + std::to_string(register_index));
    }
}

Pointer<float> State::get_second_order_weights(DendriticNode *node) const {
    try {
        return attributes.at(layer_devices.at(node->to_layer))
            .at(node->to_layer->neural_model)->get_second_order_weights(node->id);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get second order weights in State for "
            "Dendritic Node: " + node->str());
    }
}

Pointer<Output> State::get_expected(Layer *layer) const {
    try {
        return attributes.at(layer_devices.at(layer)).at(layer->neural_model)
            ->get_expected(layer->id);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get expected data in State for "
            "layer: " + layer->str());
    }
}

Pointer<Output> State::get_output(Layer *layer, int word_index) const {
    try {
        return attributes.at(layer_devices.at(layer)).at(layer->neural_model)
            ->get_output(layer->id, word_index);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get output data in State for "
            "layer: " + layer->str());
    }
}

Pointer<float> State::get_buffer_input(Layer *layer) const {
    try {
        return internal_buffers.at(layer_devices.at(layer))->get_input(layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get buffer input data in State for "
            "layer: " + layer->str());
    }
}

Pointer<Output> State::get_buffer_expected(Layer *layer) const {
    try {
        return internal_buffers.at(layer_devices.at(layer))
            ->get_expected(layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get buffer expected data in State for "
            "unrepresented layer: " + layer->str());
    }
}

DeviceID State::get_device_id(Layer *layer) const {
    try {
        return layer_devices.at(layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get device ID in State for "
            "layer: " + layer->str());
    }
}

int State::get_layer_index(Layer *layer) const {
    try {
        return attributes.at(layer_devices.at(layer)).at(layer->neural_model)
            ->get_layer_index(layer->id);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get layer index in State for "
            "layer: " + layer->str());
    }
}

int State::get_other_start_index(Layer *layer) const {
    try {
        return attributes.at(layer_devices.at(layer)).at(layer->neural_model)
            ->get_other_start_index(layer->id);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get 'other start index' in State for "
            "layer: " + layer->str());
    }
}

const Attributes* State::get_attributes_pointer(Layer *layer) const {
    try {
        return attributes.at(layer_devices.at(layer))
                         .at(layer->neural_model)->pointer;
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get Attributes pointer in State for "
            "layer: " + layer->str());
    }
}

Kernel<ATTRIBUTE_ARGS> State::get_attribute_kernel(Layer *layer) const {
    try {
        return attributes.at(layer_devices.at(layer))
                         .at(layer->neural_model)->get_kernel();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get Attribute kernel in State for "
            "layer: " + layer->str());
    }
}

Kernel<ATTRIBUTE_ARGS> State::get_learning_kernel(Layer *layer) const {
    try {
        return attributes.at(layer_devices.at(layer))
                         .at(layer->neural_model)->get_learning_kernel();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get Attribute learning kernel in State for "
            "layer: " + layer->str());
    }
}

int State::get_connection_index(Connection *conn) const {
    try {
        return attributes.at(layer_devices.at(conn->to_layer))
                         .at(conn->to_layer->neural_model)
                         ->get_connection_index(conn->id);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get connection index in State for "
            "connection: " + conn->str());
    }
}

Pointer<float> State::get_matrix(Connection* conn) const {
    try {
        return weight_matrices.at(conn)->get_data();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get weight matrix in State for "
            "conn: " + conn->str());
    }
}

EXTRACTOR State::get_connection_extractor(Connection *conn) const {
    return get_extractor(
        Attributes::get_output_type(conn->from_layer),
        layer_devices.at(conn->to_layer));
}

Kernel<SYNAPSE_ARGS> State::get_activator(Connection *conn) const {
    try {
        return attributes.at(layer_devices.at(conn->to_layer))
                         .at(conn->to_layer->neural_model)
                         ->get_activator(conn);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get activator in State for "
            "connection: " + conn->str());
    }
}

Kernel<SYNAPSE_ARGS> State::get_updater(Connection *conn) const {
    try {
        return attributes.at(layer_devices.at(conn->to_layer))
                         .at(conn->to_layer->neural_model)
                         ->get_updater(conn);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get updater in State for "
            "connection: " + conn->str());
    }
}

Pointer<Output> State::get_device_output_buffer(
        Connection *conn, int word_index) const {
    try {
        return inter_device_buffers
            .at(layer_devices.at(conn->to_layer)).at(word_index)
            ->get_output(conn->from_layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve device buffer output in State for "
            "connection: " + conn->str()
            + ", index: " + std::to_string(word_index));
    }
}

bool State::is_inter_device(Connection *conn) const {
    try {
        return layer_devices.at(conn->from_layer)
            != layer_devices.at(conn->to_layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to check inter-device status in State for "
            "connection: " + conn->str());
    }
}

BasePointer* State::get_neuron_data(Layer *layer, std::string key) {
    transfer_to_host();
    try {
        return attributes.at(layer_devices.at(layer)).at(layer->neural_model)
            ->get_neuron_data(layer->id, key);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get neuron \"" + key + "\" data in State for "
            "layer: " + layer->str());
    }
}

BasePointer* State::get_layer_data(Layer *layer, std::string key) {
    transfer_to_host();
    try {
        return attributes.at(layer_devices.at(layer)).at(layer->neural_model)
            ->get_layer_data(layer->id, key);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get layer \"" + key + "\" data in State for "
            "layer: " + layer->str());
    }
}

BasePointer* State::get_connection_data(Connection *conn, std::string key) {
    transfer_to_host();
    try {
        return attributes.at(layer_devices.at(conn->to_layer))
                         .at(conn->to_layer->neural_model)
                         ->get_connection_data(conn->id, key);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get connection \"" + key + "\" data in State for "
            "connection: " + conn->str());
    }
}

BasePointer* State::get_weight_matrix(Connection *conn) {
    transfer_to_host();
    try {
        return weight_matrices.at(conn)->get_pointer();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get weight matrix data in State for "
            "connection: " + conn->str());
    }
}

Pointer<float> State::get_weight_matrix(Connection *conn, int layer) {
    transfer_to_host();
    try {
        return weight_matrices.at(conn)->get_layer(layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get weight matrix data in State for "
            "connection: " + conn->str());
    }
}
