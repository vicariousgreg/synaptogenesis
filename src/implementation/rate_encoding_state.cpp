#include <cstdlib>
#include <cstdio>
#include <string>

#include "rate_encoding_state.h"
#include "../framework/tools.h"
#include "../framework/parallel.h"

/******************************************************************************
 **************************** INITIALIZATION **********************************
 ******************************************************************************/

static RateEncodingParameters create_parameters(std::string str) {
    return RateEncodingParameters(0.0);
    //throw ("Unrecognized parameter string: " + str).c_str();
}

void RateEncodingState::build(Model* model) {
    this->model = model;
    int num_neurons = model->num_neurons;

    this->output = (float*) allocate_host(num_neurons, sizeof(float));
    this->input = (float*) allocate_host(num_neurons, sizeof(float));
    this->neuron_parameters =
        (RateEncodingParameters*) allocate_host(num_neurons, sizeof(RateEncodingParameters));

    // Fill in table
    for (int i = 0 ; i < num_neurons ; ++i) {
        std::string &param_string = model->parameter_strings[i];
        RateEncodingParameters params = create_parameters(param_string);
        this->neuron_parameters[i] = params;
        this->input[i] = 0;
    }

#ifdef PARALLEL
    // Allocate space on GPU and copy data
    this->device_input = (float*)
        allocate_device(num_neurons, sizeof(float), this->input);
    this->device_output = (float*)
        allocate_device(num_neurons, sizeof(float), this->output);
    this->device_neuron_parameters = (RateEncodingParameters*)
        allocate_device(num_neurons, sizeof(RateEncodingParameters), this->neuron_parameters);
#endif
    this->weight_matrices = build_weight_matrices(model, 1);
}
