#include <cstring>

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

    // Distribute layers
    // Count up weights, and distribute layers in round robin fashion,
    //    in decreasing order of weights
    std::map<Layer*, int> num_weights;
    for (auto layer : model->get_layers()) {
        num_weights[layer] = 0;
        for (auto& conn : layer->get_input_connections())
            num_weights[layer] += conn->get_num_weights();
    }

    for (int i = 0; num_weights.size() > 0; ++i) {
        Layer *biggest;
        int size = -1;
        for (auto pair : num_weights) {
            if (pair.second > size) {
                size = pair.second;
                biggest = pair.first;
            }
        }
        num_weights.erase(biggest);
        layer_devices[biggest] = (i % num_devices);
    }

    // Validate neural model strings
    for (auto layer : model->get_layers())
        if (Attributes::get_neural_models().count(layer->neural_model) == 0)
            ErrorManager::get_instance()->log_error(
                "Unrecognized neural model \"" + layer->neural_model +
                "\" in layer \"" + layer->name + "\"!");

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
                auto att = build_attributes(layers, neural_model, device_id);
                attributes[device_id][neural_model] = att;

                /* Set up weight matrices */
                for (auto& layer : layers) {
                    for (auto& conn : layer->get_input_connections()) {
                        WeightMatrix* matrix = new WeightMatrix(conn,
                            att->get_matrix_depth(conn), device_id);
                        this->weight_matrices[conn] = matrix;
                        att->process_weight_matrix(matrix);
                        matrix->schedule_transfer();
                    }
                }
            }
        }
    }

    // Create buffers
    for (DeviceID i = 0; i < num_devices; ++i) {
        // Set up internal buffer
        LayerList input_layers, output_layers, expected_layers;

        // Extract layers assigned to this device
        for (auto layer : model->get_input_layers()) {
            if (layer_devices[layer] == i)
                input_layers.push_back(layer);
        }
        for (auto layer : model->get_output_layers()) {
            if (layer_devices[layer] == i)
                output_layers.push_back(layer);
        } for (auto layer : model->get_expected_layers()) {
            if (layer_devices[layer] == i)
                expected_layers.push_back(layer);
        }

        internal_buffers.push_back(
            build_buffer(i, input_layers, output_layers, expected_layers));

        // Set up inter-device buffers (one per word index)
        std::map<int, LayerList> inter_device_layers;

        // Identify any inter-device connections and make room for them
        for (auto layer : model->get_layers())
            if (layer_devices[layer] == i)
                for (auto conn : layer->get_input_connections())
                    if (is_inter_device(conn)) {
                        int word_index = get_word_index(
                            conn->delay, get_output_type(conn->from_layer));
                        inter_device_layers[word_index].push_back(conn->from_layer);
                    }

        // Create a map of word indices to buffers
        std::map<int, Buffer*> buffer_map;
        for (auto pair : inter_device_layers)
            buffer_map[pair.first] = build_buffer(
                i, LayerList(), pair.second, LayerList());

        // Set the inter device buffer
        inter_device_buffers.push_back(buffer_map);
    }

    // Finally, transfer data
    ResourceManager::get_instance()->transfer();
    for (auto n : Attributes::get_neural_models())
        for (int i = 0; i < num_devices; ++i)
            if (attributes[i][n] != nullptr)
                attributes[i][n]->transfer_to_device();
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

Pointer<float> State::get_second_order_input(DendriticNode *node) const {
    return attributes[layer_devices.at(node->to_layer)]
        .at(node->to_layer->neural_model)->get_second_order_input(node->id);
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

Kernel<SYNAPSE_ARGS> State::get_activator(
        Connection *conn, DendriticNode *node) const {
    return attributes[layer_devices.at(conn->to_layer)]
                     .at(conn->to_layer->neural_model)
                     ->get_activator(conn, node);
}

Kernel<SYNAPSE_ARGS> State::get_updater(
        Connection *conn, DendriticNode *node) const {
    return attributes[layer_devices.at(conn->to_layer)]
                     .at(conn->to_layer->neural_model)
                     ->get_updater(conn, node);
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
