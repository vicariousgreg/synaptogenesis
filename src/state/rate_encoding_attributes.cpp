#include <cstdlib>
#include <cstdio>
#include <string>

#include "state/rate_encoding_attributes.h"
#include "tools.h"
#include "parallel.h"

static RateEncodingParameters create_parameters(std::string str) {
    return RateEncodingParameters(0.0);
    //throw ("Unrecognized parameter string: " + str).c_str();
}

RateEncodingAttributes::RateEncodingAttributes(Model* model) : Attributes(model) {
    RateEncodingParameters* local_params =
        (RateEncodingParameters*) allocate_host(total_neurons, sizeof(RateEncodingParameters));

    // Fill in table
    for (int i = 0; i < model->all_layers.size(); ++i) {
        Layer *layer = model->all_layers[i];
        RateEncodingParameters params = create_parameters(layer->params);
        for (int j = 0 ; j < layer->size ; ++j)
            local_params[layer->index+j] = params;
    }

#ifdef PARALLEL
    // Allocate space on GPU and copy data
    this->neuron_parameters = (RateEncodingParameters*)
        allocate_device(total_neurons, sizeof(RateEncodingParameters), local_params);
    free(local_params);
#else
    this->neuron_parameters = local_params;
#endif
}

RateEncodingAttributes::~RateEncodingAttributes() {
#ifdef PARALLEL
    cudaFree(this->neuron_parameters);
#else
    free(this->neuron_parameters);
#endif
}
