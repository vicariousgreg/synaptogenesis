#include "state/attributes.h"
#include "state/state.h"

Attributes* Attributes::build_attributes(LayerList &layers,
        std::string neural_model, DeviceID device_id) {
    auto bank = Attributes::get_neural_model_bank();
    if (bank->neural_models.count(neural_model) == 0)
        ErrorManager::get_instance()->log_error(
            "Unrecognized neural model string: " + neural_model + "!");

    Attributes *attributes =
        bank->build_pointers[neural_model](layers);
    attributes->object_size =
        bank->sizes[neural_model];

    // Copy attributes to device and set the pointer
    attributes->set_device_id(device_id);
    return attributes;
}

Attributes::NeuralModelBank* Attributes::get_neural_model_bank() {
    static Attributes::NeuralModelBank* bank = new NeuralModelBank();
    return bank;
}

const std::set<std::string> Attributes::get_neural_models() {
    return Attributes::get_neural_model_bank()->neural_models;
}

OutputType Attributes::get_output_type() {
    return Attributes::get_neural_model_bank()
        ->output_types[get_neural_model()];
}

OutputType Attributes::get_output_type(std::string neural_model) {
    return Attributes::get_neural_model_bank()
        ->output_types[neural_model];
}

OutputType Attributes::get_output_type(Layer *layer) {
    return Attributes::get_output_type(layer->neural_model);
}

int Attributes::register_neural_model(std::string neural_model,
        int object_size, OutputType output_type, ATT_BUILD_PTR build_ptr) {
    auto bank = Attributes::get_neural_model_bank();
    if (bank->neural_models.count(neural_model) == 1)
        ErrorManager::get_instance()->log_error(
            "Duplicate neural model string: " + neural_model + "!");
    bank->neural_models.insert(neural_model);
    bank->build_pointers[neural_model] = build_ptr;
    bank->sizes[neural_model] = object_size;
    bank->output_types[neural_model] = output_type;

    // Return index as an identifier
    return bank->neural_models.size() - 1;
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
        int input_register_count = layer->dendritic_root->get_max_register_index() + 1;
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
            dendrite_DFS(layer->dendritic_root, second_order_size);
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
    this->total_layers = layers.size();;
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
    cudaFree(this->pointer);
#endif
}

int Attributes::dendrite_DFS(DendriticNode *curr, int second_order_size) {
    if (curr->is_second_order()) {
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

void Attributes::set_device_id(DeviceID device_id) {
    this->device_id = device_id;

    // Retrieve extractor
    // This has to wait until device_id is set
    get_extractor(&this->extractor, get_output_type(), device_id);
}

void Attributes::transfer_to_device() {
#ifdef __CUDACC__
    // Copy attributes to device and set the pointer
    if (not ResourceManager::get_instance()->is_host(device_id)) {
        cudaSetDevice(device_id);

        // If already transfered, free old copy
        if (this->pointer != this) cudaFree(this->pointer);

        // Transfer to device
        this->pointer = (Attributes*)
            ResourceManager::get_instance()->allocate_device(
                1, object_size, this, device_id);
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
            layer_sizes[pair.first], pair.second)] = &input;
    for (auto pair : output_start_indices)
        pointers[PointerKey(
            pair.first, "output",
            layer_sizes[pair.first], pair.second)] = &output;
    for (auto pair : other_start_indices)
        pointers[PointerKey(
            pair.first, "expected",
            layer_sizes[pair.first], pair.second)] = &expected;
    for (auto var_pair : neuron_variables)
        for (auto l_pair : other_start_indices)
            pointers[PointerKey(
                l_pair.first, var_pair.first,
                layer_sizes[l_pair.first], l_pair.second)] = var_pair.second;
    return pointers;
}

void Attributes::register_neuron_variable(std::string key, BasePointer* ptr) {
    if (this->neuron_variables.count(key) > 0)
        ErrorManager::get_instance()->log_error(
            "Repeated neuron variable key: " + key);
    this->neuron_variables[key] = ptr;
}

void Attributes::register_connection_variable(std::string key, BasePointer* ptr) {
    if (this->connection_variables.count(key) > 0)
        ErrorManager::get_instance()->log_error(
            "Repeated connection variable key: " + key);
    this->connection_variables[key] = ptr;
}

void Attributes::register_layer_variable(std::string key, BasePointer* ptr) {
    if (this->layer_variables.count(key) > 0)
        ErrorManager::get_instance()->log_error(
            "Repeated layer variable key: " + key);
    this->layer_variables[key] = ptr;
}

int Attributes::get_layer_index(size_t id) const {
    return layer_indices.at(id);
}

int Attributes::get_other_start_index(size_t id) const {
    return other_start_indices.at(id);
}

Pointer<float> Attributes::get_input(size_t id, int register_index) const {
    int size = layer_sizes.at(id);
    return input.slice(input_start_indices.at(id) + (register_index * size), size);
}

Pointer<float> Attributes::get_second_order_weights(size_t id) const {
    int size = second_order_sizes.at(id);
    return second_order_weights.slice(second_order_indices.at(id), size);
}

Pointer<Output> Attributes::get_expected(size_t id) const {
    int size = layer_sizes.at(id);
    return expected.slice(other_start_indices.at(id), size);
}

Pointer<Output> Attributes::get_output(size_t id, int word_index) const {
    int size = layer_sizes.at(id);
    return output.slice(output_start_indices.at(id) + (word_index * size), size);
}

int Attributes::get_connection_index(size_t id) const {
    return connection_indices.at(id);
}
