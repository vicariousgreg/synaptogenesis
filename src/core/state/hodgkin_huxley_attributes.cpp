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

HodgkinHuxleyAttributes::HodgkinHuxleyAttributes(Model* model)
        : SpikingAttributes(model) {
    this->h = (float*) allocate_host(total_neurons, sizeof(float));
    this->m = (float*) allocate_host(total_neurons, sizeof(float));
    this->n = (float*) allocate_host(total_neurons, sizeof(float));
    this->current_trace = (float*) allocate_host(total_neurons, sizeof(float));
    this->neuron_parameters = (HodgkinHuxleyParameters*)
        allocate_host(total_neurons, sizeof(HodgkinHuxleyParameters));

    // Fill in table
    for (auto& layer : model->get_layers()) {
        HodgkinHuxleyParameters params = create_parameters(layer->params);
        for (int j = 0 ; j < layer->size ; ++j) {
            int start_index = layer->get_start_index();
            neuron_parameters[start_index+j] = params;
            voltage[start_index+j] = -64.9997224337;
            h[start_index+j] = 0.596111046355;
            m[start_index+j] = 0.0529342176209;
            n[start_index+j] = 0.31768116758;
            current_trace[start_index+j] = 0.0;
        }
    }
}

HodgkinHuxleyAttributes::~HodgkinHuxleyAttributes() {
#ifdef PARALLEL
    cudaFree(this->h);
    cudaFree(this->m);
    cudaFree(this->n);
    cudaFree(this->current_trace);
    cudaFree(this->neuron_parameters);
#else
    free(this->h);
    free(this->m);
    free(this->n);
    free(this->current_trace);
    free(this->neuron_parameters);
#endif
}

#ifdef PARALLEL
void HodgkinHuxleyAttributes::send_to_device() {
    SpikingAttributes::send_to_device();

    // Allocate space on GPU and copy data
    float* device_h = (float*)
        allocate_device(total_neurons, sizeof(float), this->h);
    float* device_m = (float*)
        allocate_device(total_neurons, sizeof(float), this->m);
    float* device_n = (float*)
        allocate_device(total_neurons, sizeof(float), this->n);
    float* device_current_trace = (float*)
        allocate_device(total_neurons, sizeof(float), this->current_trace);
    HodgkinHuxleyParameters* device_params = (HodgkinHuxleyParameters*)
        allocate_device(total_neurons, sizeof(HodgkinHuxleyParameters),
        this->neuron_parameters);

    free(this->h);
    free(this->m);
    free(this->n);
    free(this->current_trace);
    free(this->neuron_parameters);

    this->h = device_h;
    this->m = device_m;
    this->n = device_n;
    this->current_trace = device_current_trace;
    this->neuron_parameters = device_params;
}
#endif
