#include "state/attributes.h"
#include "state/izhikevich_attributes.h"
#include "state/rate_encoding_attributes.h"
#include "engine/kernel/kernel.h"
#include "util/tools.h"
#include "util/parallel.h"

Attributes *build_attributes(Model *model) {
    Attributes *attributes;
    if (model->engine_name == "izhikevich")
        attributes = new IzhikevichAttributes(model);
    else if (model->engine_name == "rate_encoding")
        attributes = new RateEncodingAttributes(model);
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognized engine type!");
    return attributes;
}

Attributes::Attributes(Model *model, OutputType output_type)
        : output_type(output_type) {
    // Get neuron counts
    this->total_neurons = model->num_neurons;

    // Determine start indices and number of neurons for each type
    int curr_index = 0;
    for (int layer_type = 0; layer_type < IO_TYPE_SIZE; ++layer_type) {
        std::vector<Layer*> layers = model->layers[layer_type];
        int size = 0;
        for (int i = 0; i < layers.size(); ++i)
            size += layers[i]->size;

        this->start_indices[layer_type] = curr_index;
        this->num_neurons[layer_type] = size;
        curr_index += size;
    }

    // TODO: determine how many input cells are needed
    //   based on the dendritic trees of the layers

    // Allocate space for input and output
    float* local_input = (float*) allocate_host(
        this->total_neurons, sizeof(float));
    Output* local_output = (Output*) allocate_host(
        this->total_neurons * HISTORY_SIZE, sizeof(Output));

    // Retrieve attribute kernel
    get_attribute_kernel(&this->attribute_kernel, model->engine_name);

    // Retrieve extractor
    get_extractor(&this->extractor, output_type);

#ifdef PARALLEL
    // Copy data to device, then free from host
    this->input = (float*)
        allocate_device(this->total_neurons, sizeof(float), local_input);
    this->output = (Output*)
        allocate_device(this->total_neurons * HISTORY_SIZE, sizeof(Output), local_output);
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
#else
    free(this->input);
    free(this->output);
#endif
}
