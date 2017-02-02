#include "state/attributes.h"
#include "state/izhikevich_attributes.h"
#include "state/rate_encoding_attributes.h"
#include "state/hodgkin_huxley_attributes.h"
#include "engine/kernel/kernel.h"
#include "util/tools.h"
#include "util/parallel.h"

Attributes *build_attributes(Model *model) {
    Attributes *attributes;
    int object_size;

    if (model->engine_name == "izhikevich") {
        attributes = new IzhikevichAttributes(model);
        object_size = sizeof(IzhikevichAttributes);
    } else if (model->engine_name == "rate_encoding") {
        attributes = new RateEncodingAttributes(model);
        object_size = sizeof(RateEncodingAttributes);
    } else if (model->engine_name == "hodgkin_huxley") {
        attributes = new HodgkinHuxleyAttributes(model);
        object_size = sizeof(HodgkinHuxleyAttributes);
    } else {
        ErrorManager::get_instance()->log_error(
            "Unrecognized engine type!");
    }

#ifdef PARALLEL
    // Copy attributes to device and set the pointer
    // Superclass will set its own pointer otherwise
    attributes->pointer = (Attributes*)
        allocate_device(1, object_size, attributes);
#endif
    return attributes;
}

Attributes::Attributes(Model *model, OutputType output_type)
        : output_type(output_type),
          total_neurons(model->get_num_neurons()),
          pointer(this) {
    // Determine start indices and number of neurons for each type
    int curr_index = 0;
    for (auto layer_type : IOTypes) {
        auto layers = model->get_layers(layer_type);
        int size = 0;
        for (int i = 0; i < layers.size(); ++i)
            size += layers[i]->size;

        this->start_indices[layer_type] = curr_index;
        this->num_neurons[layer_type] = size;
        curr_index += size;
    }

    // Determine how many input cells are needed
    //   based on the dendritic trees of the layers
    int max_input_registers = 1;
    for (auto& layer : model->get_layers()) {
        int register_count = layer->dendritic_root->get_max_register_index() + 1;
        if (register_count > max_input_registers)
            max_input_registers = register_count;
    }

    // Allocate space for input and output
    float* local_input = (float*) allocate_host(
        this->total_neurons * max_input_registers, sizeof(float));
    Output* local_output = (Output*) allocate_host(
        this->total_neurons * HISTORY_SIZE, sizeof(Output));

    // Retrieve attribute kernel
    get_attribute_kernel(&this->attribute_kernel, model->engine_name);

    // Retrieve extractor
    get_extractor(&this->extractor, output_type);

#ifdef PARALLEL
    // Copy data to device, then free from host
    this->input = (float*)
        allocate_device(this->total_neurons * max_input_registers,
                        sizeof(float), local_input);
    this->output = (Output*)
        allocate_device(this->total_neurons * HISTORY_SIZE,
                        sizeof(Output), local_output);
    free(local_input);
    free(local_output);
#else
    // Simply set pointers
    this->input = local_input;
    this->output = local_output;
#endif
    // Create pointer to most recent word of output
    this->recent_output = this->output + ((HISTORY_SIZE-1) * this->total_neurons);
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
