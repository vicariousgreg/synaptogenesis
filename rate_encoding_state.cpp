#include <cstdlib>
#include <cstdio>
#include <string>

#include "rate_encoding_state.h"
#include "rate_encoding_operations.h"
#include "model.h"
#include "tools.h"
#include "constants.h"
#include "parallel.h"

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
        std::string &param_string = model->neuron_parameters[i];
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

/******************************************************************************
 ************************ TIMESTEP DYNAMICS ***********************************
 ******************************************************************************/

void RateEncodingState::step_input() {
    // For each weight matrix...
    //   Update Currents using synaptic input
    //     current = operation ( current , dot ( spikes * weights ) )
    for (int cid = 0 ; cid < this->model->num_connections; ++cid) {
#ifdef PARALLEL
        re_update_inputs(
            this->model->connections[cid], this->weight_matrices[cid],
            this->device_output, this->device_input, this->model->num_neurons);
#else
        re_update_inputs(
            this->model->connections[cid], this->weight_matrices[cid],
            this->output, this->input, this->model->num_neurons);
#endif
    }
}

void RateEncodingState::step_output() {
#ifdef PARALLEL
    re_update_outputs(
        this->device_output,
        this->device_input,
        this->device_neuron_parameters,
        this->model->num_neurons);
#else
    re_update_outputs(
        this->output,
        this->input,
        this->neuron_parameters,
        this->model->num_neurons);
#endif
}

void RateEncodingState::step_weights() {
    re_update_weights();
}
