#include "state/attributes.h"
#include "state/state.h"

OutputType Attributes::get_output_type() {
    return NeuralModelBank::get_output_type(get_neural_model());
}

OutputType Attributes::get_output_type(std::string neural_model) {
    return NeuralModelBank::get_output_type(neural_model);
}

OutputType Attributes::get_output_type(Layer *layer) {
    return NeuralModelBank::get_output_type(layer->neural_model);
}

Attributes::Attributes(LayerList &layers, OutputType output_type)
        : output_type(output_type),
          device_id(0),
          pointer(this) {
    // Keep track of register sizes
    int input_size = 0;
    int output_size = 0;
    int other_size = 0;
    int second_order_size = 0;

    int timesteps_per_output = get_timesteps_per_output(output_type);

    // Determine how many input cells are needed
    //   based on the dendritic trees of the layers
    for (auto& layer : layers) {
        layer_sizes[layer->id] = layer->size;
        int input_register_count = layer->get_dendritic_root()->get_max_register_index() + 1;
        int output_register_count = 1 +
            (layer->get_max_delay() / timesteps_per_output);

        // Set start indices and update size
        input_start_indices[layer->id] = input_size;
        input_size += input_register_count * layer->size;

        // Set start index, shift by layer size multiplied by
        //   number of registers for connection delays
        output_start_indices[layer->id] = output_size;
        output_size += output_register_count * layer->size;

        // Add other (expected uses this too)
        other_start_indices[layer->id] = other_size;
        other_size += layer->size;

        // Find any second order dendritic nodes
        second_order_size =
            dendrite_DFS(layer->get_dendritic_root(), second_order_size);
    }

    // Set layer indices
    int layer_index = 0;
    for (auto& layer : layers)
        layer_indices[layer->id] = layer_index++;

    // Set connection indices
    int conn_index = 0;
    for (auto& layer : layers)
        for (auto& conn : layer->get_input_connections())
            connection_indices[conn->id] = conn_index++;

    // Allocate space for input and output
    this->input = Pointer<float>(input_size, 0.0);
    this->output = Pointer<Output>(output_size);
    this->expected = Pointer<Output>(other_size);
    this->second_order_weights = Pointer<float>(second_order_size, 0.0);
    this->total_neurons = other_size;
    this->total_layers = layers.size();
    this->total_connections = get_num_connections(layers);
}

Attributes::~Attributes() {
    this->input.free();
    this->output.free();
    this->expected.free();
    this->second_order_weights.free();
    for (auto pair : neuron_variables) pair.second->free();
    for (auto pair : connection_variables) pair.second->free();
    for (auto pair : layer_variables) pair.second->free();

#ifdef __CUDACC__
    if (this->pointer != pointer and
            not ResourceManager::get_instance()->is_host(device_id)) {
        cudaFree(this->pointer);
    }
#endif
}

int Attributes::dendrite_DFS(const DendriticNode *curr, int second_order_size) {
    if (curr->second_order) {
        second_order_indices[curr->id] = second_order_size;
        int size = curr->get_second_order_size();
        second_order_sizes[curr->id] = size;
        second_order_size += size;
    } else {
        // Recurse on internal children
        for (auto& child : curr->get_children())
            if (not child->is_leaf())
                second_order_size =
                    this->dendrite_DFS(child, second_order_size);
    }
    return second_order_size;
}

void Attributes::transfer_to_device() {
#ifdef __CUDACC__
    // Copy attributes to device and set the pointer
    if (not ResourceManager::get_instance()->is_host(device_id)) {
        cudaSetDevice(device_id);

        // If already transfered, free old copy
        if (this->pointer != this)
            cudaFree(this->pointer);

        // Transfer to device
        this->pointer = (Attributes*)
            ResourceManager::get_instance()->allocate_device(
                1, get_object_size(), this, device_id);
    }
#endif
}

std::vector<BasePointer*> Attributes::get_pointers() {
    std::vector<BasePointer*> pointers = {
        &input, &output, &expected, &second_order_weights
    };
    for (auto pair : neuron_variables) pointers.push_back(pair.second);
    for (auto pair : connection_variables) pointers.push_back(pair.second);
    for (auto pair : layer_variables) pointers.push_back(pair.second);
    return pointers;
}

std::map<PointerKey, BasePointer*> Attributes::get_pointer_map() {
    std::map<PointerKey, BasePointer*> pointers;

    for (auto pair : input_start_indices)
        pointers[PointerKey(
            pair.first, "input",
            layer_sizes[pair.first] * input.get_unit_size(), pair.second)] = &input;
    for (auto pair : output_start_indices)
        pointers[PointerKey(
            pair.first, "output",
            layer_sizes[pair.first] * output.get_unit_size(), pair.second)] = &output;
    for (auto pair : other_start_indices)
        pointers[PointerKey(
            pair.first, "expected",
            layer_sizes[pair.first] * expected.get_unit_size(), pair.second)] = &expected;
    for (auto pair : second_order_indices)
        pointers[PointerKey(
            pair.first, "second order weights",
            layer_sizes[pair.first] * input.get_unit_size(), pair.second)] = &second_order_weights;
    for (auto var_pair : neuron_variables)
        for (auto l_pair : other_start_indices)
            pointers[PointerKey(
                l_pair.first, var_pair.first,
                layer_sizes[l_pair.first] * var_pair.second->get_unit_size(),
                l_pair.second)] = var_pair.second;
    return pointers;
}

template Pointer<float> Attributes::create_neuron_variable();
template Pointer<float> Attributes::create_connection_variable();
template Pointer<float> Attributes::create_layer_variable();
template Pointer<int> Attributes::create_neuron_variable();
template Pointer<int> Attributes::create_connection_variable();
template Pointer<int> Attributes::create_layer_variable();

template Pointer<float> Attributes::create_neuron_variable(float val);
template Pointer<float> Attributes::create_connection_variable(float val);
template Pointer<float> Attributes::create_layer_variable(float val);
template Pointer<int> Attributes::create_neuron_variable(int val);
template Pointer<int> Attributes::create_connection_variable(int val);
template Pointer<int> Attributes::create_layer_variable(int val);

template<class T>
Pointer<T> Attributes::create_neuron_variable() {
    return Pointer<T>(total_neurons);
}

template<class T>
Pointer<T> Attributes::create_connection_variable() {
    return Pointer<T>(total_connections);
}

template<class T>
Pointer<T> Attributes::create_layer_variable() {
    return Pointer<T>(total_layers);
}

template<class T>
Pointer<T> Attributes::create_neuron_variable(T val) {
    return Pointer<T>(total_neurons, val);
}

template<class T>
Pointer<T> Attributes::create_connection_variable(T val) {
    return Pointer<T>(total_connections, val);
}

template<class T>
Pointer<T> Attributes::create_layer_variable(T val) {
    return Pointer<T>(total_layers, val);
}

void Attributes::register_neuron_variable(
        std::string key, BasePointer* ptr) {
    if (this->neuron_variables.count(key) > 0)
        LOG_ERROR(
            "Repeated neuron variable key: " + key);
    this->neuron_variables[key] = ptr;
}

void Attributes::register_connection_variable(
        std::string key, BasePointer* ptr) {
    if (this->connection_variables.count(key) > 0)
        LOG_ERROR(
            "Repeated connection variable key: " + key);
    this->connection_variables[key] = ptr;
}

void Attributes::register_layer_variable(
        std::string key, BasePointer* ptr) {
    if (this->layer_variables.count(key) > 0)
        LOG_ERROR(
            "Repeated layer variable key: " + key);
    this->layer_variables[key] = ptr;
}

BasePointer* Attributes::get_neuron_data(size_t id, std::string key) {
    try {
        return neuron_variables.at(key)->slice(
            get_other_start_index(id), layer_sizes.at(id));
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve neuron data \"" + key + "\" in Attributes for "
            "layer ID: " + std::to_string(id));
    }
}

BasePointer* Attributes::get_layer_data(size_t id, std::string key) {
    try {
        return layer_variables.at(key)->slice(get_layer_index(id), 1);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve layer data \"" + key + "\" in Attributes for "
            "layer ID: " + std::to_string(id));
    }
}

BasePointer* Attributes::get_connection_data(size_t id, std::string key) {
    try {
        return connection_variables.at(key)->slice(get_connection_index(id), 1);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve connection data \"" + key + "\" in Attributes "
            "for connection ID: " + std::to_string(id));
    }
}


int Attributes::get_layer_index(size_t id) const {
    try {
        return layer_indices.at(id);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve layer index in Attributes for "
            "layer ID: " + std::to_string(id));
    }
}

int Attributes::get_other_start_index(size_t id) const {
    try {
        return other_start_indices.at(id);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve 'other start index' in Attributes for "
            "layer ID: " + std::to_string(id));
    }
}

Pointer<float> Attributes::get_input(size_t id, int register_index) const {
    try {
        int size = layer_sizes.at(id);
        return input.slice(
            input_start_indices.at(id) + (register_index * size), size);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve input data in Attributes for "
            "layer ID: " + std::to_string(id)
            + ", index: " + std::to_string(register_index));
    }
}

Pointer<float> Attributes::get_second_order_weights(size_t id) const {
    try {
        int size = second_order_sizes.at(id);
        return second_order_weights.slice(second_order_indices.at(id), size);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve second order weights in Attributes for "
            "layer ID: " + std::to_string(id));
    }
}

Pointer<Output> Attributes::get_expected(size_t id) const {
    try {
        int size = layer_sizes.at(id);
        return expected.slice(other_start_indices.at(id), size);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve expected data in Attributes for "
            "layer ID: " + std::to_string(id));
    }
}

Pointer<Output> Attributes::get_output(size_t id, int word_index) const {
    try {
        int size = layer_sizes.at(id);
        return output.slice(
            output_start_indices.at(id) + (word_index * size), size);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve output data in Attributes for "
            "layer ID: " + std::to_string(id)
            + ", index: " + std::to_string(word_index));
    }
}

int Attributes::get_connection_index(size_t id) const {
    try {
        return connection_indices.at(id);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve connection index in Attributes for "
            "connection ID: " + std::to_string(id));
    }
}
