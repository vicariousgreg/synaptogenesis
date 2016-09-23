#include <cstdlib>
#include <ctime>
#include <iostream>
#include <climits>
#include <stdio.h>
#include <math.h>

#include "environment.h"
#include "neuron_parameters.h"
#include "weight_matrix.h"
#include "tools.h"
#include "operations.h"

#ifdef parallel

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#endif

Environment::Environment () {
    this->num_neurons = 0;
    this->num_layers = 0;
    this->num_connections = 0;
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
    int offset = this->layers[layer_id].start_index;
    int size = this->layers[layer_id].size;
    this->state.set_current(offset, size, input);
}

void Environment::inject_random_current(int layer_id, float max) {
    int offset = this->layers[layer_id].start_index;
    int size = this->layers[layer_id].size;
    this->state.randomize_current(offset, size, max);
}

void Environment::clear_current(int layer_id) {
    int offset = this->layers[layer_id].start_index;
    int size = this->layers[layer_id].size;
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
void Environment::build() {
    int count = this->num_neurons;

    // Build weight matrices
    for (int i = 0 ; i < this->num_connections ; ++i)
        this->connections[i].build();

#ifdef parallel
    cudaMalloc(&this->neuron_parameters, count * sizeof(NeuronParameters));
    NeuronParameters *temp_params =
        (NeuronParameters*)malloc(count * sizeof(NeuronParameters));

    // Fill in table
    for (int i = 0 ; i < count ; ++i) {
        temp_params[i] = this->parameters_vector[i].copy();
    }

    // Copy values to GPU
    cudaDeviceSynchronize();
    cudaCheckError();

    cudaMemcpy(&this->neuron_parameters, temp_params,
        count * sizeof(NeuronParameters), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaCheckError();

    // Free up temporary memory
    free(temp_params);

#else
    // Then initialize actual arrays
    this->neuron_parameters = (NeuronParameters*)malloc(count * sizeof(NeuronParameters));

    // Fill in table
    for (int i = 0 ; i < count ; ++i) {
        this->neuron_parameters[i] = this->parameters_vector[i].copy();
    }
#endif

    this->state.build(count, this->neuron_parameters);
}

/*
 * Connects two layers with a weight matrix.
 * If the layer is plastic, it will learn.
 */
int Environment::connect_layers(int from_layer, int to_layer,
        bool plastic, float max_weight) {
    this->connections.push_back(WeightMatrix(
        this->layers[from_layer], this->layers[to_layer],
        plastic, max_weight));
    return this->num_connections++;
}

/*
 * Adds a layer to the environment.
 *     Adds the appropriate number of neurons according to the given size.
 *     Neurons are initialized with given parameters a,b,c,d.
 *     |sign| indicates whether the layer is excitatory or inhibitory.
 * Returns the layer's index.
 * TODO: Add more parameters
 */
int Environment::add_layer(int size, int sign,
        float a, float b, float c, float d) {
    // Index of first neuron for layer
    int start_index = this->num_neurons;
    int layer_index = this->num_layers++;

    this->layers.push_back(Layer(start_index, size, sign));

    // Add neurons.
    for (int i = 0; i < size; ++i) {
        this->add_neuron(a,b,c,d);
    }

    return layer_index;
}

/*
 */
int Environment::add_randomized_layer(
        int size, int sign) {
    // Index of first neuron for layer
    int start_index = this->num_neurons;
    int layer_index = this->num_layers++;

    this->layers.push_back(Layer(start_index, size, sign));

    // Add neurons.
    if (sign > 0) {
        for (int i = 0; i < size; ++i) {
            // (ai; bi) = (0:02; 0:2) and (ci; di) = (-65; 8) + (15;-6)r2
            float a = 0.02;
            float b = 0.2; // increase for higher frequency oscillations

            float rand = fRand(0, 1);
            float c = -65.0 + (15.0 * rand * rand);

            rand = fRand(0, 1);
            float d = 8.0 - (6.0 * (rand * rand));
            this->add_neuron(a,b,c,d);
        }
    } else {
        for (int i = 0; i < size; ++i) {
            //(ai; bi) = (0:02; 0:25) + (0:08;-0:05)ri and (ci; di)=(-65; 2).
            float rand = fRand(0, 1);
            float a = 0.02 + (0.08 * rand);

            rand = fRand(0, 1);
            float b = 0.25 - (0.05 * rand);

            float c = -65.0;
            float d = 2.0;
            this->add_neuron(a,b,c,d);
        }
    }

    return layer_index;
}

// Adds a neuron to the environment.
// Returns the neuron's index.
int Environment::add_neuron(float a, float b, float c, float d) {
    this->parameters_vector.push_back(NeuronParameters(a,b,c,d));
    return this->num_neurons++;
}

/******************************************************************************
 ************************ TIMESTEP DYNAMICS ***********************************
 ******************************************************************************/

/*
 * Performs a timestep cycle.
 */
void Environment::cycle() {
    this->activate();
#ifdef parallel
    cudaDeviceSynchronize();
    cudaCheckError();
#endif

    this->update_voltages();
#ifdef parallel
    cudaDeviceSynchronize();
    cudaCheckError();
#endif

    this->timestep();
#ifdef parallel
    cudaDeviceSynchronize();
    cudaCheckError();
#endif

    this->update_weights();
#ifdef parallel
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
    for (int cid = 0 ; cid < this->num_connections; ++cid) {
        WeightMatrix &conn = this->connections[cid];
#ifdef parallel
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
#ifdef parallel
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
#ifdef parallel
    int threads = 32;
    int blocks = ceil((float)(this->num_neurons) / threads);
    izhikevich<<<blocks, threads>>>(
#else
    izhikevich(
#endif
        this->state.voltage,
        this->state.recovery,
        this->state.current,
        this->neuron_parameters,
        this->num_neurons);
}

/*
 * Perform timestep cycling.
 * Fills the spike buffer based on voltages and the SPIKE_THRESH.
 * Increments the ages of last spikes, and resets recovery if spiked.
 */
void Environment::timestep() {
    /* 4. Timestep */
#ifdef parallel
    int threads = 32;
    int blocks = ceil((float)(this->num_neurons) / threads);
    calc_spikes<<<blocks, threads>>>(
#else
    calc_spikes(
#endif
        this->state.spikes,
        this->state.voltage,
        this->state.recovery,
        this->neuron_parameters,
        this->num_neurons);
}

/**
 * Updates weights.
 * TODO: implement.
 */
void Environment::update_weights() {
    /* 5. Update weights */
}
