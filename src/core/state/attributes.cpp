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

Attributes::Attributes(Layer *layer, OutputType output_type)
        : layer(layer),
          output_type(output_type),
          device_id(ResourceManager::get_instance()->get_host_id()),
          pointer(this) {
    // Determine how many input cells are needed
    //   based on the dendritic trees of the layers
    this->input_register_count =
        layer->get_dendritic_root()->get_max_register_index() + 1;
    this->output_register_count =
        1 + get_word_index(layer->get_max_delay(), output_type);

    this->input = Pointer<float>(input_register_count * layer->size, 0.0);
    this->expected = Pointer<Output>(layer->size);
    this->output = Pointer<Output>(output_register_count * layer->size);

    // Set up weight matrices
    for (auto& conn : layer->get_input_connections()) {
        WeightMatrix* matrix =
            NeuralModelBank::build_weight_matrix(conn);
        this->weight_matrices[conn] = matrix;
    }
}

Attributes::~Attributes() {
    this->input.free();
    this->output.free();
    this->expected.free();
    for (auto pair : neuron_variables) pair.second->free();
    for (auto matrix : weight_matrices) delete matrix.second;

    // Free device copy if one exists
    if (this != this->pointer)
        free_pointer(this->pointer, device_id);
}

void Attributes::transfer(DeviceID new_device) {
    if (device_id == new_device) return;

    auto host_id = ResourceManager::get_instance()->get_host_id();
    if (device_id != host_id and new_device != host_id)
        LOG_ERROR("Cannot transfer attributes directly between devices!");

    bool to_host = (new_device == host_id);
    auto old_device = device_id;
    auto old_pointer = this->pointer;

    // Transfer data and update pointer
    if (to_host) {
        transfer_pointer(old_pointer, this,
            get_object_size(),
            device_id, new_device);
        this->pointer = this;
    } else {
        this->pointer = (Attributes*)
            transfer_pointer(this, nullptr,
                get_object_size(),
                device_id, new_device);
    }

    // Update device ID
    this->device_id = new_device;

    // If old copy was not on the host, free it
    if (old_device != host_id)
        free_pointer(old_pointer, old_device);

    // Transfer weight matrices
    for (auto pair : weight_matrices)
        pair.second->transfer(new_device);
}

std::vector<BasePointer*> Attributes::get_pointers() {
    std::vector<BasePointer*> pointers;
    for (auto pair : get_pointer_map())
        pointers.push_back(pair.second);
    return pointers;
}

std::map<PointerKey, BasePointer*> Attributes::get_pointer_map() {
    std::map<PointerKey, BasePointer*> pointers;

    pointers[PointerKey(
        layer->id, "input", input.get_bytes())] = &input;
    pointers[PointerKey(
        layer->id, "output", output.get_bytes())] = &output;
    pointers[PointerKey(
        layer->id, "expected", expected.get_bytes())] = &expected;
    for (auto pair : neuron_variables)
        pointers[PointerKey(
            layer->id, pair.first,
            pair.second->get_bytes())] = pair.second;

    for (auto wm_pair : weight_matrices)
        for (auto pair : wm_pair.second->get_pointer_map())
            if (pointers.count(pair.first) > 0)
                LOG_ERROR("Duplicate PointerKey encountered in Attributes!");
            else pointers[pair.first] = pair.second;
    return pointers;
}

void Attributes::process_weight_matrices() {
    for (auto pair : weight_matrices)
        process_weight_matrix(pair.second);
}

void Attributes::transpose_weight_matrices() {
    for (auto pair : weight_matrices)
        pair.second->transpose();
}

template Pointer<float> Attributes::create_neuron_variable();
template Pointer<int> Attributes::create_neuron_variable();

template Pointer<float> Attributes::create_neuron_variable(float val);
template Pointer<int> Attributes::create_neuron_variable(int val);

template<class T>
Pointer<T> Attributes::create_neuron_variable() {
    return Pointer<T>(layer->size);
}

template<class T>
Pointer<T> Attributes::create_neuron_variable(T val) {
    return Pointer<T>(layer->size, val);
}

void Attributes::register_neuron_variable(
        std::string key, BasePointer* ptr) {
    if (this->neuron_variables.count(key) > 0)
        LOG_ERROR("Repeated neuron variable key: " + key);
    this->neuron_variables[key] = ptr;
}

BasePointer* Attributes::get_neuron_data(std::string key) {
    try {
        return neuron_variables.at(key);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve neuron data \"" + key + "\" in Attributes!");
    }
}

Pointer<float> Attributes::get_input(int register_index) const {
    try {
        return input.slice(register_index * layer->size, layer->size);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve input data in Attributes for index: "
            + std::to_string(register_index));
    }
}

Pointer<Output> Attributes::get_expected() const {
    return expected;
}

Pointer<Output> Attributes::get_output(int word_index) const {
    try {
        return output.slice(word_index * layer->size, layer->size);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve output data in Attributes for index:"
            + std::to_string(word_index));
    }
}
