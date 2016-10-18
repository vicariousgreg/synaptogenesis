#include <cstdlib>
#include <cstdio>
#include <string>

#include "state/rate_encoding_state.h"
#include "tools.h"
#include "parallel.h"

/******************************************************************************
 **************************** INITIALIZATION **********************************
 ******************************************************************************/

static RateEncodingParameters create_parameters(std::string str) {
    return RateEncodingParameters(0.0);
    //throw ("Unrecognized parameter string: " + str).c_str();
}

RateEncodingState::RateEncodingState(Model* model) : State(model) {
    float* local_output = (float*) allocate_host(num_neurons, sizeof(float));
    float* local_input = (float*) allocate_host(num_neurons, sizeof(float));
    RateEncodingParameters* local_params =
        (RateEncodingParameters*) allocate_host(num_neurons, sizeof(RateEncodingParameters));

    // Fill in table
    for (int i = 0 ; i < num_neurons ; ++i) {
        std::string &param_string = model->parameter_strings[i];
        RateEncodingParameters params = create_parameters(param_string);
        local_params[i] = params;
        local_input[i] = 0;
    }

#ifdef PARALLEL
    // Allocate space on GPU and copy data
    this->input = (float*)
        allocate_device(num_neurons, sizeof(float), local_input);
    this->output = (float*)
        allocate_device(num_neurons, sizeof(float), local_output);
    this->neuron_parameters = (RateEncodingParameters*)
        allocate_device(num_neurons, sizeof(RateEncodingParameters), local_params);
#else
    this->input = local_input;
    this->output = local_output;
    this->neuron_parameters = local_params;
#endif
    this->weight_matrices = build_weight_matrices(model, 1);
}

RateEncodingState::~RateEncodingState() {
#ifdef PARALLEL
    cudaFree(this->neuron_parameters);
    cudaFree(this->output);
#else
    free(this->neuron_parameters);
    free(this->output);
#endif
}
