#include <cstring>
#include <fstream>
#include <sys/stat.h>

#include "state/state.h"
#include "state/weight_matrix.h"
#include "state/neural_model_bank.h"
#include "network/network.h"
#include "io/buffer.h"

static std::map<Layer*, DeviceID> distribute_layers(
        const LayerList& layers, std::set<DeviceID> devices) {
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
    std::map<DeviceID, int> device_weights;
    for (auto device : devices)
        device_weights[device] = 0;

    // Give the next biggest layer to the device with the least weight
    //   until no layers are left to distribute
    for (int i = 0; num_weights.size() > 0; ++i) {
        // Start with the max device ID
        // If multi-GPU and an imbalanced network, this can shift the
        //   burden off of the the primary GPU
        int next_device = *devices.begin();
        for (auto pair : device_weights)
            if (pair.first > next_device)
                next_device = pair.first;

        for (auto pair : device_weights)
            if (pair.second < device_weights.at(next_device))
                next_device = pair.first;

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
    // Validate neural model strings
    auto neural_models = NeuralModelBank::get_neural_models();
    for (auto layer : network->get_layers())
        if (neural_models.count(layer->neural_model) == 0)
            LOG_ERROR(
                "Unrecognized neural model \"" + layer->neural_model +
                "\" in " + layer->str());

    // Create attributes and weight matrices
    for (auto layer : network->get_layers()) {
        auto att = NeuralModelBank::build_attributes(layer);
        if (not att->check_compatibility(layer->structure->cluster_type))
            LOG_ERROR(
                "Error building attributes for " + layer->str() + ":\n"
                "  Cluster compatibility conflict detected!");
        attributes[layer] = att;
        att->process_weight_matrices();
    }

    // Adjust sparse indicies
    // If not sparse, this is a no-op
    // Otherwise, it ensures weight indices are valid
    for (auto conn : network->get_connections())
        attributes.at(conn->to_layer)
            ->get_weight_matrix(conn)
            ->adjust_sparse_indices();

    this->build();
}

void State::build(std::set<DeviceID> devices) {
    // If no devices are specified, just use host
    if (devices.size() == 0)
        devices = { ResourceManager::get_instance()->get_host_id() };

    // Check if the state is already on the specified devices
    if (devices == active_devices) return;

    // Ensure the state is back on the host before rebuilding
    if (not on_host) this->transfer_to_host();

    // Distribute layers
    this->active_devices = devices;
    this->layer_devices = distribute_layers(network->get_layers(), devices);

    // Delete old buffers
    for (auto pair : internal_buffers)
        delete pair.second;
    internal_buffers.clear();
    for (auto pair : inter_device_buffers)
        for (auto inner_pair : pair.second)
            delete inner_pair.second;
    inter_device_buffers.clear();

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
                        inter_device_layers[word_index].push_back(
                            conn->from_layer);
                    }

        // Create a map of word indices to buffers
        std::map<int, Buffer*> buffer_map;
        for (auto pair : inter_device_layers)
            buffer_map[pair.first] = build_buffer(
                device_id, LayerList(), pair.second, LayerList());

        // Set the inter device buffer
        inter_device_buffers[device_id] = buffer_map;
    }
}

State::~State() {
    for (auto pair : attributes) delete pair.second;
    for (auto buffer : internal_buffers) delete buffer.second;
    for (auto map : inter_device_buffers)
        for (auto pair : map.second)
            delete pair.second;
}

PointerMap State::get_pointer_map() const {
    PointerMap pointer_map;

    for (auto att_pair : attributes) {
        for (auto pair : att_pair.second->get_pointer_map())
            if (pointer_map.count(pair.first) > 0)
                LOG_ERROR("Duplicate PointerKey encountered in State!");
            else pointer_map[pair.first] = pair.second;
    }

    return pointer_map;
}

PointerSetMap State::get_network_pointers() const {
    PointerSetMap network_pointers;
    for (auto device : active_devices)
        network_pointers[device] = std::vector<BasePointer*>();

    for (auto pair : layer_devices)
        for (auto ptr : attributes.at(pair.first)->get_pointers())
            network_pointers[pair.second].push_back(ptr);

    return network_pointers;
}

PointerSetMap State::get_buffer_pointers() const {
    PointerSetMap buffer_pointers;
    for (auto device : active_devices)
        buffer_pointers[device] = std::vector<BasePointer*>();

    for (auto pair : internal_buffers)
        for (auto ptr : pair.second->get_pointers())
            buffer_pointers[pair.first].push_back(ptr);

    for (auto pair : inter_device_buffers)
        for (auto inner_pair : pair.second)
            for (auto ptr : inner_pair.second->get_pointers())
                buffer_pointers[pair.first].push_back(ptr);

    return buffer_pointers;
}

void State::transfer_to_device() {
    if (not on_host) return;

    auto res_man = ResourceManager::get_instance();
    DeviceID host_id = res_man->get_host_id();

    // If the host is the only active device, return
    if (active_devices.size() == 1
            and *active_devices.begin() == host_id)
        return;

    // Transfer network pointers
    for (auto pair : get_network_pointers())
        for (auto ptr : pair.second)
            ptr->transfer(pair.first);

    // Transfer buffer pointers
    for (auto pair : get_buffer_pointers())
        for (auto ptr : pair.second)
            ptr->transfer(pair.first);

    // Transfer attributes
    for (auto pair : attributes)
        pair.second->transfer(layer_devices.at(pair.first));

    // Transpose matrices
    for (auto pair : attributes)
        if (pair.second->get_device_id() != host_id)
            pair.second->transpose_weight_matrices();

    on_host = false;
}

void State::transfer_to_host() {
    if (on_host) return;

    auto res_man = ResourceManager::get_instance();
    DeviceID host_id = res_man->get_host_id();

    // If the host is the only active device, return
    if (active_devices.size() == 1
            and *active_devices.begin() == host_id)
        return;

    // Transpose matrices
    for (auto pair : attributes)
        if (pair.second->get_device_id() != host_id)
            pair.second->transpose_weight_matrices();

    // Transfer attributes
    for (auto pair : attributes)
        pair.second->transfer(host_id);

    // Transfer buffer pointers
    for (auto pair : get_buffer_pointers())
        for (auto ptr : pair.second)
            ptr->transfer(host_id);

    // Transfer network pointers
    for (auto pair : get_network_pointers())
        for (auto ptr : pair.second)
            ptr->transfer(host_id);

    on_host = true;
}

void State::copy_to(State* other) {
    // Transfer states to the host if necessary
    if (not this->on_host) this->transfer_to_host();
    if (not other->on_host) other->transfer_to_host();

    // Copy over any matching pointers
    auto this_pointer_map = this->get_pointer_map();
    auto other_pointer_map = other->get_pointer_map();

    for (auto pair : this_pointer_map) {
        if (other_pointer_map.count(pair.first) > 0) {
            auto this_ptr = this_pointer_map.at(pair.first);
            auto other_ptr = other_pointer_map.at(pair.first);
            if (this_ptr->get_bytes() == other_ptr->get_bytes())
                this_ptr->copy_to(other_ptr);
        }
    }
}

size_t State::get_network_bytes() const {
    size_t size = 0;
    for (auto pair : get_network_pointers())
        for (auto ptr : pair.second)
            size += ptr->get_bytes();
    return size;
}

size_t State::get_buffer_bytes() const {
    size_t size = 0;
    for (auto pair : get_buffer_pointers())
        for (auto ptr : pair.second)
            size += ptr->get_bytes();
    return size;
}

bool State::exists(std::string file_name) {
    std::ifstream f((file_name).c_str());
    return f.good();
}

void State::save(std::string file_name, bool verbose) {
    // Transfer to host
    this->transfer_to_host();

    // Open file stream
    std::string path = file_name;
    std::ofstream output_file(path, std::ofstream::binary);
    if (verbose)
        printf("Saving network state to %s ...\n", path.c_str());

    for (auto pair : get_pointer_map()) {
        size_t bytes = pair.first.bytes;
        if (bytes > 0) {
            const char* data = (const char*)pair.second->get();

            if (not output_file.write(
                    (const char*)&pair.first, sizeof(PointerKey))
                or not output_file.write(data, bytes))
                LOG_ERROR(
                    "Error writing state to file!");
        }
    }

    // Close file stream
    output_file.close();
}

void State::load(std::string file_name, bool verbose) {
    if (not State::exists(file_name))
        LOG_ERROR("Could not open file: " + file_name);

    // Open file stream
    std::string path = file_name;
    std::ifstream input_file(path, std::ifstream::binary);
    if (verbose)
        printf("Loading network state from %s ...\n", path.c_str());

    // Transfer to host
    this->transfer_to_host();

    // Determine file length
    input_file.seekg (0, input_file.end);
    auto length = input_file.tellg();
    input_file.seekg (0, input_file.beg);

    unsigned long read = 0;
    while (read < length) {
        PointerKey key(0,0,0);
        if (not input_file.read((char*)&key, sizeof(PointerKey)))
            LOG_ERROR(
                "Error reading pointer key from file!");

        // Search for the pointer, and if not found, skip it
        try {
            BasePointer* ptr = get_pointer_map().at(key);

            // If pointer size doesn't match, resize
            if (ptr->get_bytes() != key.bytes)
                ptr->resize(key.bytes / ptr->get_unit_size());

            // Read the data into memory
            if (not input_file.read((char*)ptr->get(), key.bytes))
                LOG_ERROR("Error reading data from file!");

            read += sizeof(PointerKey) + key.bytes;
        } catch (...) {
            LOG_WARNING(
                "Error retrieving pointer -- continuing...");
            input_file.ignore(key.bytes);
            read += sizeof(PointerKey) + key.bytes;
        }
    }

    // Resize the weight matrices
    // This updates num_weights for sparse matrices
    for (auto pair : attributes)
        pair.second->resize_weight_matrices();

    // Close file stream
    input_file.close();
}

/* Zoo of Getters */
Pointer<float> State::get_input(Layer *layer, int register_index) const {
    try {
        return attributes.at(layer)->get_input(register_index);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get input data in State for "
            "layer: " + layer->str()
            + ", index: " + std::to_string(register_index));
    }
}

Pointer<float> State::get_second_order_weights(DendriticNode *node) const {
    try {
        return attributes.at(node->to_layer)
            ->get_weight_matrix(node->get_second_order_connection())
            ->get_second_order_weights();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get second order weights in State for "
            "Dendritic Node: " + node->str());
    }
}

Pointer<Output> State::get_expected(Layer *layer) const {
    try {
        return attributes.at(layer)->get_expected();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get expected data in State for "
            "layer: " + layer->str());
    }
}

Pointer<Output> State::get_output(Layer *layer, int word_index) const {
    // If -1, use last index (most recent output)
    while (word_index < 0)
        word_index = attributes.at(layer)->output_register_count + word_index;

    try {
        return attributes.at(layer)->get_output(word_index);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get output data in State for "
            "layer: " + layer->str());
    }
}

Pointer<float> State::get_buffer_input(Layer *layer) const {
    try {
        return
            internal_buffers.at(layer_devices.at(layer)) ->get_input(layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get buffer input data in State for "
            "layer: " + layer->str());
    }
}

Pointer<Output> State::get_buffer_expected(Layer *layer) const {
    try {
        return
            internal_buffers.at(layer_devices.at(layer))->get_expected(layer);
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

const Attributes* State::get_attributes_pointer(Layer *layer) const {
    try {
        return attributes.at(layer)->pointer;
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get Attributes pointer in State for "
            "layer: " + layer->str());
    }
}

Kernel<ATTRIBUTE_ARGS> State::get_attribute_kernel(Layer *layer) const {
    try {
        return attributes.at(layer)->get_kernel();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get Attribute kernel in State for "
            "layer: " + layer->str());
    }
}

Kernel<ATTRIBUTE_ARGS> State::get_learning_kernel(Layer *layer) const {
    try {
        return attributes.at(layer)->get_learning_kernel();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get Attribute learning kernel in State for "
            "layer: " + layer->str());
    }
}

Pointer<float> State::get_weights(Connection* conn) const {
    try {
        return attributes.at(conn->to_layer)
            ->get_weight_matrix(conn)->get_weights();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get weight matrix in State for "
            "conn: " + conn->str());
    }
}

const WeightMatrix* State::get_matrix(Connection* conn) const {
    try {
        return attributes.at(conn->to_layer)->get_weight_matrix(conn);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get weight matrix in State for "
            "conn: " + conn->str());
    }
}

const WeightMatrix* State::get_matrix_pointer(Connection* conn) const {
    try {
        return attributes.at(conn->to_layer)->get_weight_matrix(conn)->pointer;
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

AGGREGATOR State::get_connection_aggregator(Connection *conn) const {
    return get_aggregator(
        conn->opcode,
        layer_devices.at(conn->to_layer));
}

KernelList<SYNAPSE_ARGS> State::get_activators(Connection *conn) const {
    try {
        return attributes.at(conn->to_layer)->get_activators(conn);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get activator in State for "
            "connection: " + conn->str());
    }
}

KernelList<SYNAPSE_ARGS> State::get_updaters(Connection *conn) const {
    try {
        return attributes.at(conn->to_layer)->get_updaters(conn);
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

bool State::get_transpose_flag(Connection *conn) const {
    try {
        return attributes.at(conn->to_layer)
            ->get_weight_matrix(conn)
            ->get_transpose_flag();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to check transpose flag in State for "
            "connection: " + conn->str());
    }
}

BasePointer* State::get_neuron_data(Layer *layer, std::string key) {
    transfer_to_host();

    try {
        return attributes.at(layer)->get_neuron_data(key);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get neuron \"" + key + "\" data in State for "
            "layer: " + layer->str());
    }
}

BasePointer* State::get_layer_data(Layer *layer, std::string key) {
    transfer_to_host();
    // TODO
    return nullptr;

    /*
    try {
        return attributes.at(layer)->get_layer_data(layer->id, key);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get layer \"" + key + "\" data in State for "
            "layer: " + layer->str());
    }
    */
}

BasePointer* State::get_connection_data(Connection *conn, std::string key) {
    transfer_to_host();
    // TODO
    return nullptr;

    /*
    try {
        return attributes.at(layer)->get_connection_data(conn->id, key);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get connection \"" + key + "\" data in State for "
            "connection: " + conn->str());
    }
    */
}

BasePointer* State::get_weight_matrix(Connection *conn, std::string key) {
    transfer_to_host();
    try {
        return attributes.at(conn->to_layer)
            ->get_weight_matrix(conn)
            ->get_layer(key);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to get weight matrix data in State for "
            "connection: " + conn->str());
    }
}
