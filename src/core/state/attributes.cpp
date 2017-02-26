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
    // Keep track of register sizes
    std::vector<int> register_sizes;

    // Determine how many input cells are needed
    //   based on the dendritic trees of the layers
    for (auto& layer : structure->get_layers()) {
        sizes[layer->id] = layer->size;
        int register_count = layer->dendritic_root->get_max_register_index() + 1;

        // Ensure enough sizes exist
        while (register_count > register_sizes.size()) {
            register_sizes.push_back(0);
            start_indices.push_back(std::map<int,int>());
        }

        // Set start indices and update size
        for (int i = 0; i < register_count; ++i) {
            start_indices[i][layer->id] = register_sizes[i];
            register_sizes[i] += layer->size;
        }
    }

    // Allocate space for input and output
    for (int i = 0; i < register_sizes.size(); ++i)
        this->input_registers.push_back(Pointer<float>(register_sizes[i], 0.0));
    this->input = this->input_registers[0];
    this->output = Pointer<Output>(this->total_neurons * HISTORY_SIZE);

    // Retrieve extractor
    get_extractor(&this->extractor, output_type);
}

Attributes::~Attributes() {
    for (auto& ptr : this->input_registers) ptr.free();
    this->output.free();
#ifdef PARALLEL
    cudaFree(this->pointer);
#endif
}

void Attributes::transfer_to_device() {
    // Transfer data
    for (auto& ptr : this->input_registers) ptr.transfer_to_device();
    this->input = this->input_registers[0];
    this->output.transfer_to_device();
}

int Attributes::get_start_index(int id, int register_index) const {
    return start_indices[register_index].at(id);
}

Pointer<float> Attributes::get_input(int id, int register_index) const {
    return input_registers[register_index].slice(
        start_indices[register_index].at(id),
        sizes.at(id));
}

Pointer<Output> Attributes::get_output(int id, int word_index) const {
    if (word_index >= HISTORY_SIZE)
        ErrorManager::get_instance()->log_error(
            "Cannot retrieve output word index past history length!");
    return output.slice(
        (total_neurons * word_index) + start_indices[0].at(id), sizes.at(id));
}
