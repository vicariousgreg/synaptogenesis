#include <cstdlib>
#include <cstdio>
#include <string>

#include "state/rate_encoding_attributes.h"
#include "util/tools.h"
#include "util/error_manager.h"
#include "util/parallel.h"

static RateEncodingParameters create_parameters(std::string str) {
    return RateEncodingParameters(0.0);

    //ErrorManager::get_instance()->log_error(
    //    "Unrecognized parameter string: " + str);
}

RateEncodingAttributes::RateEncodingAttributes(Model* model) : Attributes(model, FLOAT) {
    RateEncodingParameters* local_params =
        (RateEncodingParameters*) allocate_host(total_neurons, sizeof(RateEncodingParameters));

    // Fill in table
    for (int i = 0; i < model->all_layers.size(); ++i) {
        Layer *layer = model->all_layers[i];
        RateEncodingParameters params = create_parameters(layer->params);
        for (int j = 0 ; j < layer->size ; ++j)
            local_params[layer->index+j] = params;
    }

    this->input = this->get_input();
    this->output = (float*)this->get_output();
    this->recent_output = (float*)this->get_recent_output();

#ifdef PARALLEL
    // Allocate space on GPU and copy data
    this->neuron_parameters = (RateEncodingParameters*)
        allocate_device(total_neurons, sizeof(RateEncodingParameters), local_params);
    free(local_params);

    // Copy this to device
    this->device_pointer = (RateEncodingAttributes*)
        allocate_device(1, sizeof(RateEncodingAttributes), this);
#else
    this->neuron_parameters = local_params;
#endif
}

RateEncodingAttributes::~RateEncodingAttributes() {
#ifdef PARALLEL
    cudaFree(this->neuron_parameters);
    cudaFree(this->device_pointer);
#else
    free(this->neuron_parameters);
#endif
}
