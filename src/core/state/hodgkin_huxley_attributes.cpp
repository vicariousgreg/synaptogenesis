#include <string>
#include <sstream>

#include "state/hodgkin_huxley_attributes.h"
#include "util/tools.h"
#include "util/error_manager.h"
#include "util/parallel.h"

static HodgkinHuxleyParameters create_parameters(std::string str) {
    std::stringstream stream(str);
    float iapp;
    stream >> iapp;
    return HodgkinHuxleyParameters(iapp);
}

HodgkinHuxleyAttributes::HodgkinHuxleyAttributes(Model* model) : Attributes(model, BIT) {
    float* local_voltage = (float*) allocate_host(total_neurons, sizeof(float));
    float* local_h = (float*) allocate_host(total_neurons, sizeof(float));
    float* local_m = (float*) allocate_host(total_neurons, sizeof(float));
    float* local_n = (float*) allocate_host(total_neurons, sizeof(float));
    float* local_current_trace = (float*) allocate_host(total_neurons, sizeof(float));
    HodgkinHuxleyParameters* local_params =
        (HodgkinHuxleyParameters*) allocate_host(total_neurons, sizeof(HodgkinHuxleyParameters));

    // Fill in table
    for (int i = 0; i < model->all_layers.size(); ++i) {
        Layer *layer = model->all_layers[i];
        HodgkinHuxleyParameters params = create_parameters(layer->params);
        for (int j = 0 ; j < layer->size ; ++j) {
            local_params[layer->start_index+j] = params;
            local_voltage[layer->start_index+j] = -64.9997224337;
            local_h[layer->start_index+j] = 0.596111046355;
            local_m[layer->start_index+j] = 0.0529342176209;
            local_n[layer->start_index+j] = 0.31768116758;
            local_current_trace[layer->start_index+j] = 0.0;
        }
    }

    this->current = this->input;
    this->spikes = (unsigned int*)this->output;

#ifdef PARALLEL
    // Allocate space on GPU and copy data
    this->voltage = (float*)
        allocate_device(total_neurons, sizeof(float), local_voltage);
    this->h = (float*)
        allocate_device(total_neurons, sizeof(float), local_h);
    this->m = (float*)
        allocate_device(total_neurons, sizeof(float), local_m);
    this->n = (float*)
        allocate_device(total_neurons, sizeof(float), local_n);
    this->current_trace = (float*)
        allocate_device(total_neurons, sizeof(float), local_current_trace);
    this->neuron_parameters = (HodgkinHuxleyParameters*)
        allocate_device(total_neurons, sizeof(HodgkinHuxleyParameters), local_params);
    free(local_voltage);
    free(local_h);
    free(local_m);
    free(local_n);
    free(local_current_trace);
    free(local_params);

    // Copy this to device
    this->device_pointer = (HodgkinHuxleyAttributes*)
        allocate_device(1, sizeof(HodgkinHuxleyAttributes), this);
#else
    this->voltage = local_voltage;
    this->h = local_h;
    this->m = local_m;
    this->n = local_n;
    this->current_trace = local_current_trace;
    this->neuron_parameters = local_params;
#endif
}

HodgkinHuxleyAttributes::~HodgkinHuxleyAttributes() {
#ifdef PARALLEL
    cudaFree(this->voltage);
    cudaFree(this->h);
    cudaFree(this->m);
    cudaFree(this->n);
    cudaFree(this->current_trace);
    cudaFree(this->neuron_parameters);
    cudaFree(this->device_pointer);
#else
    free(this->voltage);
    free(this->h);
    free(this->m);
    free(this->n);
    free(this->current_trace);
    free(this->neuron_parameters);
#endif
}
