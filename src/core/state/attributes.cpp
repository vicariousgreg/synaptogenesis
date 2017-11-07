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
          device_id(ResourceManager::get_instance()->get_host_id()),
          pointer(this) {
    // Keep track of register sizes
    int input_size = 0;
    int output_size = 0;
    int other_size = 0;

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
    }

    // Set layer indices
    int layer_index = 0;
    for (auto& layer : layers)
        layer_indices[layer->id] = layer_index++;

    // Allocate space for input and output
    this->input = Pointer<float>(input_size, 0.0);
    this->output = Pointer<Output>(output_size);
    this->expected = Pointer<Output>(other_size);
    this->total_neurons = other_size;
    this->total_layers = layers.size();
    this->total_connections = get_num_connections(layers);
}

Attributes::~Attributes() {
    this->input.free();
    this->output.free();
    this->expected.free();
    for (auto pair : neuron_variables) pair.second->free();
    for (auto pair : layer_variables) pair.second->free();

#ifdef __CUDACC__
    if (this != this->pointer and
            not ResourceManager::get_instance()->is_host(device_id)) {
        cudaSetDevice(device_id);
        cudaFree(this->pointer);
    }
#endif
}

void Attributes::transfer_to_device() {
#ifdef __CUDACC__
    // Copy attributes to device and set the pointer
    if (not ResourceManager::get_instance()->is_host(device_id)) {
        cudaSetDevice(device_id);

        // If already transferred, free old copy
        if (this->pointer != this)
            cudaFree(this->pointer);

        // Transfer to device
        this->pointer = (Attributes*)
            ResourceManager::get_instance()->allocate_device(
                1, get_object_size(), this, device_id);
    }
#endif
}

void Attributes::transfer_to_host() {
#ifdef __CUDACC__
    // Copy attributes to device and set the pointer
    if (not ResourceManager::get_instance()->is_host(device_id)
            and this != this->pointer) {
        cudaSetDevice(device_id);

        // Transfer to host
        cudaMemcpy(this, this->pointer, get_object_size(), cudaMemcpyDeviceToHost);

        // If previously transferred, free old copy
        if (this->pointer != this)
            cudaFree(this->pointer);
        this->pointer = this;
    }
#endif
}

std::vector<BasePointer*> Attributes::get_pointers() {
    std::vector<BasePointer*> pointers = {
        &input, &output, &expected
    };
    for (auto pair : neuron_variables) pointers.push_back(pair.second);
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
    for (auto var_pair : neuron_variables)
        for (auto l_pair : other_start_indices)
            pointers[PointerKey(
                l_pair.first, var_pair.first,
                layer_sizes[l_pair.first] * var_pair.second->get_unit_size(),
                l_pair.second)] = var_pair.second;
    return pointers;
}

template Pointer<float> Attributes::create_neuron_variable();
template Pointer<float> Attributes::create_layer_variable();
template Pointer<int> Attributes::create_neuron_variable();
template Pointer<int> Attributes::create_layer_variable();

template Pointer<float> Attributes::create_neuron_variable(float val);
template Pointer<float> Attributes::create_layer_variable(float val);
template Pointer<int> Attributes::create_neuron_variable(int val);
template Pointer<int> Attributes::create_layer_variable(int val);

template<class T>
Pointer<T> Attributes::create_neuron_variable() {
    return Pointer<T>(total_neurons);
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
