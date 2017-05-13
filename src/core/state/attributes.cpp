#include "state/attributes.h"
#include "state/state.h"

Attributes *build_attributes(LayerList &layers,
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

int Attributes::register_neural_model(std::string neural_model,
        int object_size, BUILD_PTR build_ptr) {
    auto bank = Attributes::get_neural_model_bank();
    if (bank->neural_models.count(neural_model) == 1)
        ErrorManager::get_instance()->log_error(
            "Duplicate neural model string: " + neural_model + "!");
    bank->neural_models.insert(neural_model);
    bank->build_pointers[neural_model] = build_ptr;
    bank->sizes[neural_model] = object_size;

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
    int expected_size = 0;
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

        if (layer->is_expected()) {
            expected_start_indices[layer->id] = expected_size;
            expected_size += layer->size;
        }

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
    this->expected = Pointer<Output>(expected_size);
    this->second_order_input = Pointer<float>(second_order_size, 0.0);
    this->total_neurons = other_size;
    this->total_layers = layers.size();;
}

Attributes::~Attributes() {
    this->input.free();
    this->output.free();
    this->expected.free();
    this->second_order_input.free();
    for (auto ptr : managed_variables) ptr->free();

#ifdef __CUDACC__
    cudaFree(this->pointer);
#endif
}

int Attributes::dendrite_DFS(DendriticNode *curr, int second_order_size) {
    auto res_man = ResourceManager::get_instance();

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
    get_extractor(&this->extractor, output_type, device_id);
}

void Attributes::transfer_to_device() {
#ifdef __CUDACC__
    // Copy attributes to device and set the pointer
    if (not ResourceManager::get_instance()->is_host(device_id)) {
        // If already transfered, free old copy
        cudaSetDevice(device_id);
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
        &input, &output, &expected, &second_order_input
    };
    for (auto ptr : managed_variables)
        pointers.push_back(ptr);
    return pointers;
}

void Attributes::register_variable(BasePointer* ptr) {
    this->managed_variables.push_back(ptr);
}

int Attributes::get_layer_index(int id) const {
    return layer_indices.at(id);
}

int Attributes::get_other_start_index(int id) const {
    return other_start_indices.at(id);
}

Pointer<float> Attributes::get_input(int id, int register_index) const {
    int size = layer_sizes.at(id);
    return input.slice(input_start_indices.at(id) + (register_index * size), size);
}

Pointer<float> Attributes::get_second_order_input(int id) const {
    int size = second_order_sizes.at(id);
    return second_order_input.slice(second_order_indices.at(id), size);
}

Pointer<Output> Attributes::get_expected(int id) const {
    int size = layer_sizes.at(id);
    return expected.slice(expected_start_indices.at(id), size);
}

Pointer<Output> Attributes::get_output(int id, int word_index) const {
    int size = layer_sizes.at(id);
    return output.slice(output_start_indices.at(id) + (word_index * size), size);
}

int Attributes::get_connection_index(int id) const {
    return connection_indices.at(id);
}
