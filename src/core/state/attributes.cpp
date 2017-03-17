#include "state/attributes.h"
#include "state/izhikevich_attributes.h"
#include "state/rate_encoding_attributes.h"
#include "state/hodgkin_huxley_attributes.h"

Attributes *build_attributes(LayerList &layers,
        NeuralModel neural_model, DeviceID device_id) {
    Attributes *attributes;

    switch(neural_model) {
        case(IZHIKEVICH):
            attributes = new IzhikevichAttributes(layers);
            attributes->object_size = sizeof(IzhikevichAttributes);
            break;
        case(HODGKIN_HUXLEY):
            attributes = new HodgkinHuxleyAttributes(layers);
            attributes->object_size = sizeof(HodgkinHuxleyAttributes);
            break;
        case(RATE_ENCODING):
            attributes = new RateEncodingAttributes(layers);
            attributes->object_size = sizeof(RateEncodingAttributes);
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "Unrecognized engine type!");
    }

    // Copy attributes to device and set the pointer
    attributes->set_device_id(device_id);
    attributes->schedule_transfer();
    return attributes;
}

Attributes::Attributes(LayerList &layers, OutputType output_type,
        Kernel<ATTRIBUTE_ARGS> kernel)
        : output_type(output_type),
          kernel(kernel),
          device_id(0),
          pointer(this) {
    // Keep track of register sizes
    int input_size = 0;
    int output_size = 0;
    int expected_size = 0;
    int other_size = 0;

    int timesteps_per_output = get_timesteps_per_output(output_type);

    // Determine how many input cells are needed
    //   based on the dendritic trees of the layers
    for (auto& layer : layers) {
        sizes[layer->id] = layer->size;
        int register_count = layer->dendritic_root->get_max_register_index() + 1;

        // Determine max delay for output connections
        int max_delay_registers = 0;
        for (auto& conn : layer->get_output_connections()) {
            int delay_registers = conn->delay / timesteps_per_output;
            if (delay_registers > max_delay_registers)
                max_delay_registers = delay_registers;
        }
        ++max_delay_registers;


        // Set start indices and update size
        input_start_indices[layer->id] = input_size;
        input_size += register_count * layer->size;

        output_start_indices[layer->id] = output_size;
        output_size += max_delay_registers * layer->size;

        if (layer->is_expected()) {
            expected_start_indices[layer->id] = expected_size;
            expected_size += layer->size;
        }

        other_start_indices[layer->id] = other_size;
        other_size += layer->size;
    }

    // Allocate space for input and output
    this->input = Pointer<float>(input_size, 0.0);
    this->output = Pointer<Output>(output_size);
    this->expected = Pointer<Output>(expected_size);
    this->total_neurons = other_size;
}

Attributes::~Attributes() {
    this->input.free();
    this->output.free();
    this->expected.free();
#ifdef __CUDACC__
    cudaFree(this->pointer);
#endif
}

void Attributes::transfer_to_device() {
#ifdef __CUDACC__
    // Copy attributes to device and set the pointer
    if (not ResourceManager::get_instance()->is_host(device_id))
        this->pointer = (Attributes*)
            ResourceManager::get_instance()->allocate_device(
                1, object_size, this, device_id);
#endif
}

void Attributes::schedule_transfer() {
    // Transfer data
    this->input.schedule_transfer(device_id);
    this->output.schedule_transfer(device_id);
    this->expected.schedule_transfer(device_id);
}

int Attributes::get_other_start_index(int id) const {
    return other_start_indices.at(id);
}

Pointer<float> Attributes::get_input(int id, int register_index) const {
    int size = sizes.at(id);
    return input.slice(input_start_indices.at(id) + (register_index * size), size);
}

Pointer<Output> Attributes::get_expected(int id) const {
    int size = sizes.at(id);
    return expected.slice(expected_start_indices.at(id), size);
}

Pointer<Output> Attributes::get_output(int id, int word_index) const {
    int size = sizes.at(id);
    return output.slice(output_start_indices.at(id) + (word_index * size), size);
}
