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

Attributes::Attributes(Structure *structure, OutputType output_type)
        : output_type(output_type),
          total_neurons(structure->get_num_neurons()),
          pointer(this) {
    // Determine start indices for each layer
    int curr_index = 0;
    for (auto& layer : structure->get_layers()) {
        start_indices[layer->id] = curr_index;
        sizes[layer->id] = layer->size;
        curr_index += layer->size;
    }

    // Determine how many input cells are needed
    //   based on the dendritic trees of the layers
    max_input_registers = 1;
    for (auto& layer : structure->get_layers()) {
        int register_count = layer->dendritic_root->get_max_register_index() + 1;
        if (register_count > max_input_registers)
            max_input_registers = register_count;
    }

    // Allocate space for input and output
    this->input = new Pointer<float>(this->total_neurons * max_input_registers);
    this->output = new Pointer<Output>(this->total_neurons * HISTORY_SIZE);

    // Retrieve extractor
    get_extractor(&this->extractor, output_type);
}

Attributes::~Attributes() {
    delete this->input;
    delete this->output;
#ifdef PARALLEL
    cudaFree(this->pointer);
#endif
}

void Attributes::transfer_to_device() {
    // Transfer data
    this->input->transfer_to_device();
    this->output->transfer_to_device();
}

int Attributes::get_start_index(int id) const {
    return start_indices.at(id);
}

Pointer<float> *Attributes::get_input(int id) const {
    return input->splice(start_indices.at(id), sizes.at(id));
}

Pointer<Output> *Attributes::get_output(int id, int word_index) const {
    if (word_index >= HISTORY_SIZE)
        ErrorManager::get_instance()->log_error(
            "Cannot retrieve output word index past history length!");
    return output->splice(
        (total_neurons * word_index) + start_indices.at(id), sizes.at(id));
}
