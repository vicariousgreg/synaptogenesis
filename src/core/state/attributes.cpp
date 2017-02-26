#include "state/attributes.h"
#include "state/izhikevich_attributes.h"
#include "state/rate_encoding_attributes.h"
#include "state/hodgkin_huxley_attributes.h"
#include "util/tools.h"
#include "util/parallel.h"

Attributes *build_attributes(Structure *structure) {
    Attributes *attributes;
    int object_size;

    if (structure->engine_name == "izhikevich") {
        attributes = new IzhikevichAttributes(structure);
        object_size = sizeof(IzhikevichAttributes);
    } else if (structure->engine_name == "rate_encoding") {
        attributes = new RateEncodingAttributes(structure);
        object_size = sizeof(RateEncodingAttributes);
    } else if (structure->engine_name == "hodgkin_huxley") {
        attributes = new HodgkinHuxleyAttributes(structure);
        object_size = sizeof(HodgkinHuxleyAttributes);
    } else {
        ErrorManager::get_instance()->log_error(
            "Unrecognized engine type!");
    }

#ifdef PARALLEL
    // Copy attributes to device and set the pointer
    // Superclass will set its own pointer otherwise
    attributes->transfer_to_device();
    attributes->pointer = (Attributes*)
        allocate_device(1, object_size, attributes);
#endif
    return attributes;
}

Attributes::Attributes(Structure *structure, OutputType output_type,
        ATTRIBUTE_KERNEL kernel)
        : output_type(output_type),
          total_neurons(structure->get_num_neurons()),
          kernel(kernel),
          pointer(this) {
    // Keep track of register sizes
    int input_size = 0;
    int output_size = 0;
    int other_size = 0;

    int timesteps_per_output = get_timesteps_per_output(output_type);

    // Determine how many input cells are needed
    //   based on the dendritic trees of the layers
    for (auto& layer : structure->get_layers()) {
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

        other_start_indices[layer->id] = other_size;
        other_size += layer->size;
    }

    // Allocate space for input and output
    this->input = Pointer<float>(input_size, 0.0);
    this->output = Pointer<Output>(output_size);

    // Retrieve extractor
    get_extractor(&this->extractor, output_type);
}

Attributes::~Attributes() {
    this->input.free();
    this->output.free();
#ifdef PARALLEL
    cudaFree(this->pointer);
#endif
}

void Attributes::transfer_to_device() {
    // Transfer data
    this->input.transfer_to_device();
    this->output.transfer_to_device();
}

int Attributes::get_input_start_index(int id) const {
    return input_start_indices.at(id);
}

int Attributes::get_output_start_index(int id) const {
    return output_start_indices.at(id);
}

int Attributes::get_other_start_index(int id) const {
    return other_start_indices.at(id);
}

Pointer<float> Attributes::get_input(int id, int register_index) const {
    int size = sizes.at(id);
    return input.slice(input_start_indices.at(id) + (register_index * size), size);
}

Pointer<Output> Attributes::get_output(int id, int word_index) const {
    int size = sizes.at(id);
    return output.slice(output_start_indices.at(id) + (word_index * size), size);
}
