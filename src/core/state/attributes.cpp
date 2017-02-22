#include "state/attributes.h"
#include "state/izhikevich_attributes.h"
#include "state/rate_encoding_attributes.h"
#include "state/hodgkin_huxley_attributes.h"
#include "engine/stream_cluster.h"
#include "engine/kernel/kernel.h"
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
    attributes->send_to_device();
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
    this->input = (float*) allocate_host(
        this->total_neurons * max_input_registers, sizeof(float));
    this->output = (Output*) allocate_host(
        this->total_neurons * HISTORY_SIZE, sizeof(Output));

    // Retrieve extractor
    get_extractor(&this->extractor, output_type);
}

Attributes::~Attributes() {
#ifdef PARALLEL
    cudaFree(this->input);
    cudaFree(this->output);
    cudaFree(this->pointer);
#else
    free(this->input);
    free(this->output);
#endif
}

StreamCluster *Attributes::build_stream_cluster(Structure *structure, State *state) {
    return new ParallelStreamCluster(structure, state);
}

#ifdef PARALLEL
void Attributes::send_to_device() {
    // Copy data to device, then free from host
    float* device_input = (float*)
        allocate_device(this->total_neurons * max_input_registers,
                        sizeof(float), this->input);
    Output* device_output = (Output*)
        allocate_device(this->total_neurons * HISTORY_SIZE,
                        sizeof(Output), this->output);
    free(this->input);
    free(this->output);
    this->input = device_input;
    this->output = device_output;
}
#endif

int Attributes::get_start_index(int id) const {
    return start_indices.at(id);
}

float* Attributes::get_input(int id) const {
    return input + start_indices.at(id);
}

Output* Attributes::get_output(int id, int word_index) const {
    if (word_index >= HISTORY_SIZE)
        ErrorManager::get_instance()->log_error(
            "Cannot retrieve output word index past history length!");
    return output + (total_neurons * word_index) + start_indices.at(id);
}
