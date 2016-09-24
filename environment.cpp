#include <cstdlib>
#include <ctime>
#include <iostream>
#include <climits>
#include <stdio.h>
#include <math.h>

#include "environment.h"
#include "model.h"
#include "neuron_parameters.h"
#include "weight_matrix.h"
#include "tools.h"
#include "operations.h"

#include "parallel.h"

Environment::Environment (Model model) {
    this->model = model;
    srand(time(NULL));
    //srand(0);
}

/******************************************************************************
 ************************* GETTER / SETTER ************************************
 ******************************************************************************/

int* Environment::get_spikes() {
    return this->state.get_spikes();
}

float* Environment::get_current() {
    return this->state.get_current();
}

void Environment::inject_current(int layer_id, float* input) {
    int offset = this->model.layers[layer_id].start_index;
    int size = this->model.layers[layer_id].size;
    this->state.set_current(offset, size, input);
}

void Environment::inject_random_current(int layer_id, float max) {
    int offset = this->model.layers[layer_id].start_index;
    int size = this->model.layers[layer_id].size;
    this->state.randomize_current(offset, size, max);
}

void Environment::clear_current(int layer_id) {
    int offset = this->model.layers[layer_id].start_index;
    int size = this->model.layers[layer_id].size;
    this->state.clear_current(offset, size);
}


/******************************************************************************
 ********************** INITIALIZATION FUNCTIONS ******************************
 ******************************************************************************/

/*
 * Builds the environment.
 * During dynamic construction, instantiation is lazy.
 * Neuron parameters are tracked in a vector, but the neuron attributes table
 *   is not initialized until this function is called.
 */
bool Environment::build() {
    int count = this->model.num_neurons;

    // Build weight matrices
    for (int i = 0 ; i < this->model.num_connections ; ++i) {
        WeightMatrix &conn = this->model.connections[i];
        if (!conn.build()) {
            printf("Failed to allocate %d (%d) -> %d (%d) matrix!\n",
                conn.from_index, conn.from_size, conn.to_index, conn.to_size);
            return false;
        }
    }

#ifdef PARALLEL
    cudaMalloc((void**)&this->neuron_parameters, count * sizeof(NeuronParameters));
    if (!cudaCheckError()) {
        printf("Failed to allocate memory on device for neuron parameters!\n");
        return false;
    }
    
    // Set up temporary parameters list
    NeuronParameters *temp_params =
        (NeuronParameters*)malloc(count * sizeof(NeuronParameters));
    if (!temp_params) {
        printf("Failed to allocate memory on host for neuron parameters!\n");
        return false;
    }
    for (int i = 0 ; i < count ; ++i) {
        temp_params[i] = this->model.parameters_vector[i].copy();
    }

    // Copy values to GPU
    cudaMemcpy(this->neuron_parameters, temp_params,
        count * sizeof(NeuronParameters), cudaMemcpyHostToDevice);
    cudaSync();
    if (!cudaCheckError()) {
        printf("Failed to copy neuron parameters to device!\n");
        return false;
    }

    // Free up temporary memory
    free(temp_params);

#else
    // Set up parameters list
    // New array copy for memory continuity
    this->neuron_parameters = (NeuronParameters*)malloc(count * sizeof(NeuronParameters));
    if (!this->neuron_parameters) {
        printf("Failed to allocate memory for neuron parameters!\n");
        return false;
    }
    for (int i = 0 ; i < count ; ++i) {
        this->neuron_parameters[i] = this->model.parameters_vector[i].copy();
    }
#endif
    // Build the state.
    if (!this->state.build(count, this->model.parameters_vector)) {
        printf("Failed to build environment state!\n");
        return false;
    }
    return true;
}


/******************************************************************************
 ************************ TIMESTEP DYNAMICS ***********************************
 ******************************************************************************/

/*
 * Performs a timestep cycle.
 */
void Environment::cycle() {
    this->activate();
#ifdef PARALLEL
    cudaDeviceSynchronize();
    cudaCheckError();
#endif

    this->update_voltages();
#ifdef PARALLEL
    cudaDeviceSynchronize();
    cudaCheckError();
#endif

    this->timestep();
#ifdef PARALLEL
    cudaDeviceSynchronize();
    cudaCheckError();
#endif

    this->update_weights();
#ifdef PARALLEL
    cudaDeviceSynchronize();
    cudaCheckError();
#endif
}

/*
 * Performs activation during a timestep.
 * For each weight matrix, calculate sum of inputs for each neuron
 *   and add it to the current vector.
 */
void Environment::activate() {
    float* current = this->state.current;
    int* spikes = this->state.recent_spikes;

    /* 2. Activation */
    // For each weight matrix...
    //   Update Currents using synaptic input
    //     current += sign * dot ( spikes * weights )
    //
    // TODO: optimize order, create batches of parallelizable computations,
    //       and move cuda barriers around batches
    for (int cid = 0 ; cid < this->model.num_connections; ++cid) {
        WeightMatrix &conn = this->model.connections[cid];
#ifdef PARALLEL
        int threads = 32;
        int blocks = ceil((float)(conn.to_size) / threads);
        mult<<<blocks, threads>>>(
#else
        mult(
#endif
            conn.sign,
            spikes + conn.from_index,  // only most recent
            conn.mData,
            current + conn.to_index,
            conn.from_size,
            conn.to_size);
#ifdef PARALLEL
        cudaDeviceSynchronize();
        cudaCheckError();
#endif
    }
}


/*
 * Perform voltage update according to input currents using Izhikevich
 *   with Euler's method.
 */
void Environment::update_voltages() {
    /* 3. Voltage Updates */
#ifdef PARALLEL
    int threads = 32;
    int blocks = ceil((float)(this->model.num_neurons) / threads);
    izhikevich<<<blocks, threads>>>(
#else
    izhikevich(
#endif
        this->state.voltage,
        this->state.recovery,
        this->state.current,
        this->neuron_parameters,
        this->model.num_neurons);
}

/*
 * Perform timestep cycling.
 * Fills the spike buffer based on voltages and the SPIKE_THRESH.
 * Increments the ages of last spikes, and resets recovery if spiked.
 */
void Environment::timestep() {
    /* 4. Timestep */
#ifdef PARALLEL
    int threads = 32;
    int blocks = ceil((float)(this->model.num_neurons) / threads);
    calc_spikes<<<blocks, threads>>>(
#else
    calc_spikes(
#endif
        this->state.spikes,
        this->state.voltage,
        this->state.recovery,
        this->neuron_parameters,
        this->model.num_neurons);
}

/**
 * Updates weights.
 * TODO: implement.
 */
void Environment::update_weights() {
    /* 5. Update weights */
}
