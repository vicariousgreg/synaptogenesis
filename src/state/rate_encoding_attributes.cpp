#include <cstdlib>
#include <cstdio>
#include <string>

#include "state/rate_encoding_attributes.h"
#include "engine/rate_encoding_kernel.h"
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

#ifdef PARALLEL
void RateEncodingAttributes::update(int start_index, int count, cudaStream_t &stream) {
    int threads = calc_threads(count);
    int blocks = calc_blocks(count);
    shift_output<<<blocks, threads, 0, stream>>>(
#else
void RateEncodingAttributes::update(int start_index, int count) {
    shift_output(
#endif
        (float*)output, start_index, count, total_neurons);
#ifdef PARALLEL
    cudaCheckError("Failed to update neuron output!");

    activation_function<<<blocks, threads, 0, stream>>>(
#else
    activation_function(
#endif
        (float*)recent_output,
        input,
        neuron_parameters,
        start_index, count);
#ifdef PARALLEL
    cudaCheckError("Failed to update neuron output!");
#endif
}
