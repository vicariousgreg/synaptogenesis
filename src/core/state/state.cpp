#include <cstring>
#include <fstream>
#include <sys/stat.h>

#include "state/state.h"
#include "state/weight_matrix.h"
#include "model/model.h"
#include "util/tools.h"

State::State(Model *model) : model(model) {
    // Determine number of non-host devices
    // Subtract one if cuda is enabled to avoid distribution to the host
    this->num_devices = ResourceManager::get_instance()->get_num_devices();
#ifdef __CUDACC__
    --this->num_devices;
#endif

    // Add pointer vectors for each device
    for (int i = 0 ; i < num_devices ; ++i) {
        network_pointers.push_back(std::vector<BasePointer*>());
        buffer_pointers.push_back(std::vector<BasePointer*>());
    }

    // Distribute layers
    // Count up weights
    std::map<Layer*, int> num_weights;
    for (auto layer : model->get_layers()) {
        num_weights[layer] = 0;
        for (auto& conn : layer->get_input_connections())
            num_weights[layer] += conn->get_num_weights();
    }

    // Keep track of weight distribution to devices
    std::vector<int> device_weights;
    for (int i = 0 ; i < num_devices ; ++i)
        device_weights.push_back(0);

    // Give the next biggest layer to the device with the least weight
    //   until no layers are left to distribute
    for (int i = 0; num_weights.size() > 0; ++i) {
        // Typically display device is 0, so start at the other end
        // This helps avoid burdening the display device in some situations
        int next_device = this->num_devices - 1;
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

    // Validate neural model strings
    for (auto layer : model->get_layers())
        if (Attributes::get_neural_models().count(layer->neural_model) == 0)
            ErrorManager::get_instance()->log_error(
                "Unrecognized neural model \"" + layer->neural_model +
                "\" in " + layer->str());

    // Create attributes and weight matrices
    for (DeviceID device_id = 0 ; device_id < num_devices ; ++device_id) {
        attributes.push_back(std::map<std::string, Attributes*>());
        for (auto neural_model : Attributes::get_neural_models()) {
            LayerList layers;
            for (auto layer : model->get_layers(neural_model))
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

                /* Set up weight matrices */
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
    for (DeviceID device_id = 0; device_id < num_devices; ++device_id) {
        // Set up internal buffer
        LayerList input_layers, output_layers, expected_layers;

        // Extract layers assigned to this device
        for (auto layer : model->get_input_layers()) {
            if (layer_devices[layer] == device_id)
                input_layers.push_back(layer);
        }
        for (auto layer : model->get_output_layers()) {
            if (layer_devices[layer] == device_id)
                output_layers.push_back(layer);
        } for (auto layer : model->get_expected_layers()) {
            if (layer_devices[layer] == device_id)
                expected_layers.push_back(layer);
        }

        auto buffer =
            build_buffer(device_id, input_layers, output_layers, expected_layers);
        internal_buffers.push_back(buffer);

        // Retrieve pointers
        for (auto ptr : buffer->get_pointers())
            buffer_pointers[device_id].push_back(ptr);

        // Set up inter-device buffers (one per word index)
        std::map<int, LayerList> inter_device_layers;

        // Identify any inter-device connections and make room for them
        for (auto layer : model->get_layers())
            if (layer_devices[layer] == device_id)
                for (auto conn : layer->get_input_connections())
                    if (is_inter_device(conn)) {
                        int word_index = get_word_index(
                            conn->delay, get_output_type(conn->from_layer));
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
        inter_device_buffers.push_back(buffer_map);
    }

    // Finally, transfer data
    this->transfer_to_device();
}

State::~State() {
    for (int i = 0; i < num_devices; ++i)
        for (auto neural_model : Attributes::get_neural_models())
            if (attributes[i][neural_model] != nullptr)
                delete attributes[i][neural_model];
    for (auto matrix : weight_matrices) delete matrix.second;
    for (auto buffer : internal_buffers) delete buffer;
    for (auto map : inter_device_buffers)
        for (auto pair : map)
            delete pair.second;
}

void State::transfer_to_device() {
#ifdef __CUDACC__
    auto res_man = ResourceManager::get_instance();
    DeviceID host_id = res_man->get_host_id();

    // Transfer network pointers
    for (int device_id = 0 ; device_id < network_pointers.size() ; ++device_id)
        if (device_id != host_id)
            res_man->transfer(device_id, network_pointers[device_id]);

    // Transfer buffer pointers
    for (int device_id = 0 ; device_id < buffer_pointers.size() ; ++device_id)
        if (device_id != host_id)
            res_man->transfer(device_id, buffer_pointers[device_id]);

    // Transfer attributes
    for (auto n : Attributes::get_neural_models())
        for (int i = 0; i < num_devices; ++i)
            if (attributes[i][n] != nullptr)
                attributes[i][n]->transfer_to_device();
#endif
}

void State::transfer_to_host() {
#ifdef __CUDACC__
    auto res_man = ResourceManager::get_instance();
    DeviceID host_id = res_man->get_host_id();

    // Transfer network pointers
    for (int device_id = 0 ; device_id < network_pointers.size() ; ++device_id)
        if (device_id != host_id)
            res_man->transfer(host_id, network_pointers[device_id]);

    // Transfer buffer pointers
    for (int device_id = 0 ; device_id < buffer_pointers.size() ; ++device_id)
        if (device_id != host_id)
            res_man->transfer(host_id, buffer_pointers[device_id]);
#endif
}

bool State::exists(std::string file_name) {
    std::ifstream f(("./states/" + file_name).c_str());
    return f.good();
}

void State::save(std::string file_name) {
    // Transfer to host
    this->transfer_to_host();

    // Open file stream
    std::string path = "./states/" + file_name;
    std::ofstream output_file(path, std::ofstream::binary);
    printf("Saving network state to %s ...\n", path.c_str());

    for (auto pair : pointer_map) {
        size_t bytes = pair.first.bytes;
        size_t offset = pair.first.offset;
        const char* data = (const char*)pair.second->get(offset);

        if (not output_file.write((const char*)&pair.first, sizeof(PointerKey)))
            ErrorManager::get_instance()->log_error(
                "Error writing state to file!");
        if (not output_file.write(data, bytes))
            ErrorManager::get_instance()->log_error(
                "Error writing state to file!");
    }

    // Close file stream
    output_file.close();
}

void State::load(std::string file_name) {
    // Transfer to host
    this->transfer_to_host();

    // Open file stream
    std::string path = "./states/" + file_name;
    std::ifstream input_file(path, std::ifstream::binary);
    printf("Loading network state from %s ...\n", path.c_str());

    // Determine file length
    input_file.seekg (0, input_file.end);
    auto length = input_file.tellg();
    input_file.seekg (0, input_file.beg);

    unsigned long read = 0;

    while (read < length) {
        PointerKey key(0,0,0,0);
        if (not input_file.read((char*)&key, sizeof(PointerKey)))
            ErrorManager::get_instance()->log_error(
                "Error reading pointer key from file!");

        char* ptr;
        try {
            ptr = (char*) pointer_map.at(key)->get(key.offset);
        } catch (...) {
            ErrorManager::get_instance()->log_warning(
                "Error retrieving pointer -- continuing...");
            continue;
        }

        if (not input_file.read(ptr, key.bytes))
            ErrorManager::get_instance()->log_error(
                "Error reading data from file!");

        read += sizeof(PointerKey) + key.bytes;
    }

    // Close file stream
    input_file.close();

    // Transfer to device
    this->transfer_to_device();
}

bool State::check_compatibility(Structure *structure) {
    // Check relevant attributes for compatibility
    for (auto n : Attributes::get_neural_models())
        for (int i = 0; i < num_devices; ++i)
            if (structure->contains(n) and attributes[i][n] and not
                    attributes[i][n]->check_compatibility(
                        structure->cluster_type))
                return false;
    return true;
}

/* Zoo of Getters */
Pointer<float> State::get_input(Layer *layer, int register_index) const {
    return attributes[layer_devices.at(layer)].at(layer->neural_model)
        ->get_input(layer->id, register_index);
}

Pointer<float> State::get_second_order_weights(DendriticNode *node) const {
    return attributes[layer_devices.at(node->to_layer)]
        .at(node->to_layer->neural_model)->get_second_order_weights(node->id);
}

Pointer<Output> State::get_expected(Layer *layer) const {
    return attributes[layer_devices.at(layer)].at(layer->neural_model)
        ->get_expected(layer->id);
}

Pointer<Output> State::get_output(Layer *layer, int word_index) const {
    return attributes[layer_devices.at(layer)].at(layer->neural_model)
        ->get_output(layer->id, word_index);
}

Pointer<float> State::get_buffer_input(Layer *layer) const {
    return internal_buffers.at(layer_devices.at(layer))->get_input(layer);
}

Pointer<Output> State::get_buffer_expected(Layer *layer) const {
    return internal_buffers.at(layer_devices.at(layer))->get_expected(layer);
}

Pointer<Output> State::get_buffer_output(Layer *layer) const {
    return internal_buffers.at(layer_devices.at(layer))->get_output(layer);
}

OutputType State::get_output_type(Layer *layer) const {
    return attributes[layer_devices.at(layer)]
                     .at(layer->neural_model)->output_type;
}

DeviceID State::get_device_id(Layer *layer) const {
    return layer_devices.at(layer);
}

int State::get_layer_index(Layer *layer) const {
    return attributes[layer_devices.at(layer)].at(layer->neural_model)
        ->get_layer_index(layer->id);
}

int State::get_other_start_index(Layer *layer) const {
    return attributes[layer_devices.at(layer)].at(layer->neural_model)
        ->get_other_start_index(layer->id);
}

const Attributes* State::get_attributes_pointer(Layer *layer) const {
    return attributes[layer_devices.at(layer)].at(layer->neural_model)->pointer;
}

Kernel<ATTRIBUTE_ARGS> State::get_attribute_kernel(Layer *layer) const {
    return attributes[layer_devices.at(layer)]
                     .at(layer->neural_model)->get_kernel();
}

Kernel<ATTRIBUTE_ARGS> State::get_learning_kernel(Layer *layer) const {
    return attributes[layer_devices.at(layer)]
                     .at(layer->neural_model)->get_learning_kernel();
}

int State::get_connection_index(Connection *conn) const {
    return attributes[layer_devices.at(conn->to_layer)]
                     .at(conn->to_layer->neural_model)
                     ->get_connection_index(conn->id);
}

Pointer<float> State::get_matrix(Connection* conn) const {
    return weight_matrices.at(conn)->get_data();
}

EXTRACTOR State::get_extractor(Connection *conn) const {
    return attributes[layer_devices.at(conn->from_layer)]
                     .at(conn->from_layer->neural_model)->extractor;
}

Kernel<SYNAPSE_ARGS> State::get_activator(Connection *conn) const {
    return attributes[layer_devices.at(conn->to_layer)]
                     .at(conn->to_layer->neural_model)
                     ->get_activator(conn);
}

Kernel<SYNAPSE_ARGS> State::get_updater(Connection *conn) const {
    return attributes[layer_devices.at(conn->to_layer)]
                     .at(conn->to_layer->neural_model)
                     ->get_updater(conn);
}

Pointer<Output> State::get_device_output_buffer(
        Connection *conn, int word_index) const {
    return inter_device_buffers
        .at(layer_devices.at(conn->to_layer)).at(word_index)
        ->get_output(conn->from_layer);
}

bool State::is_inter_device(Connection *conn) const {
    return layer_devices.at(conn->from_layer)
        != layer_devices.at(conn->to_layer);
}
