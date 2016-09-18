#include <cstdlib>
#include <ctime>
#include <iostream>
#include <climits>
#include <stdio.h>

#include "environment.h"
#include "connectivity_matrix.h"
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

/******************************************************************************
 ************************* GETTER / SETTER ************************************
 ******************************************************************************/

int* Environment::get_spikes() {
#ifdef parallel
    // Copy from GPU to local location
    cudaMemcpy(this->local_spikes, this->nat[SPIKE], this->num_neurons * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();
    return this->local_spikes;
#else
    return (int*)this->nat[SPIKE];
#endif
}

double* Environment::get_currents() {
#ifdef parallel
    // Copy from GPU to local location
    cudaMemcpy(this->local_currents, this->nat[CURRENT], this->num_neurons * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();
    return this->local_currents;
#else
    return (double*)this->nat[CURRENT];
#endif
}

void Environment::inject_random_current(int layer_id, double max) {
    int offset = this->layers[layer_id].start_index;
    int size = this->layers[layer_id].size;
#ifdef parallel
    // Send to GPU
    double* temp = (double*)malloc(size * sizeof(double));
    for (int nid = 0 ; nid < size; ++nid) {
        temp[nid] = fRand(0, max);
    }
    this->inject_current(layer_id, temp);
    free(temp);
#else
    for (int nid = offset ; nid < offset + size; ++nid) {
        ((double*)this->nat[CURRENT])[nid] = fRand(0, max);
    }
#endif
}

void Environment::inject_current(int layer_id, double* input) {
    int offset = this->layers[layer_id].start_index;
    int size = this->layers[layer_id].size;
#ifdef parallel
    // Send to GPU
    void* current = &((double*)this->nat[CURRENT])[offset];
    cudaMemcpy(current, input, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();
#else
    for (int nid = 0 ; nid < size; ++nid) {
        ((double*)this->nat[CURRENT])[nid+offset] = input[nid];
    }
#endif
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
    // Initialize Neuron Attributes Table
    // First, initialize pointers to arrays
    this->nat = (void**)malloc(SIZE * sizeof(void*));
    int count = this->num_neurons;

#ifdef parallel
    // Local spikes for output reporting
    this->local_spikes = (int*)calloc(count, sizeof(int));
    this->local_currents = (double*)calloc(count, sizeof(double));

    // Allocate space on GPU
    cudaMalloc(&this->nat[CURRENT], count * sizeof(double));
    cudaMalloc(&this->nat[VOLTAGE], count * sizeof(double));
    cudaMalloc(&this->nat[RECOVERY], count * sizeof(double));
    cudaMalloc(&this->nat[SPIKE], count * sizeof(int));
    cudaMalloc(&this->nat[AGE], count * sizeof(int));
    cudaMalloc(&this->nat[PARAMS], count * sizeof(NeuronParameters));
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();

    // Make temporary arrays for initialization
    double *temp_current = (double*)malloc(count * sizeof(double));
    double *temp_voltage = (double*)malloc(count * sizeof(double));
    double *temp_recovery = (double*)malloc(count * sizeof(double));
    int *temp_spike = (int*)malloc(count * sizeof(int));
    int *temp_age = (int*)malloc(count * sizeof(int));
    NeuronParameters *temp_params = (NeuronParameters*)malloc(count * sizeof(NeuronParameters));

    // Fill in table
    for (int i = 0 ; i < count ; ++i) {
        NeuronParameters &params = this->neuron_parameters[i];
        temp_params[i] = this->neuron_parameters[i].copy();

        temp_current[i] = 0;
        temp_voltage[i] = params.c;
        temp_recovery[i] = params.b * params.c;
        temp_spike[i] = 0;
        temp_age[i] = INT_MAX;
    }

    // Copy values to GPU
    cudaMemcpy(this->nat[CURRENT], temp_current, count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(this->nat[VOLTAGE], temp_voltage, count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(this->nat[RECOVERY], temp_recovery, count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(this->nat[SPIKE], temp_spike, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->nat[AGE], temp_age, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->nat[PARAMS], temp_params, count * sizeof(NeuronParameters), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();

    // Free up temporary memory
    free(temp_current);
    free(temp_voltage);
    free(temp_recovery);
    free(temp_spike);
    free(temp_age);
    free(temp_params);

#else
    // Then initialize actual arrays
    this->nat[CURRENT] = malloc(count * sizeof(double));
    this->nat[VOLTAGE] = malloc(count * sizeof(double));
    this->nat[RECOVERY] = malloc(count * sizeof(double));
    this->nat[SPIKE] = malloc(count * sizeof(int));
    this->nat[AGE] = malloc(count * sizeof(int));
    this->nat[PARAMS] = malloc(count * sizeof(NeuronParameters));

    // Fill in table
    for (int i = 0 ; i < count ; ++i) {
        NeuronParameters &params = this->neuron_parameters[i];
        ((NeuronParameters*)this->nat[PARAMS])[i] = this->neuron_parameters[i].copy();

        ((double*)this->nat[CURRENT])[i] = 0;
        ((double*)this->nat[VOLTAGE])[i] = params.c;
        ((double*)this->nat[RECOVERY])[i] = params.b * params.c;
        ((int*)this->nat[SPIKE])[i] = 0;
        ((int*)this->nat[AGE])[i] = INT_MAX;
    }
#endif
}

/*
 * Connects two layers with a connectivity matrix.
 * If the layer is plastic, it will learn.
 * TODO: this initializes with random weights. provide means of changing that
 */
int Environment::connect_layers(int from_layer, int to_layer,
        bool plastic, double max_weight) {
    this->connections.push_back(ConnectivityMatrix(
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
        double a, double b, double c, double d) {
    // Index of first neuron for layer
    int start_index = this->num_neurons;
    int layer_index = this->num_layers++;

    this->layers.push_back(Layer(start_index, size, layer_index, sign));

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

    this->layers.push_back(Layer(start_index, size, layer_index, sign));

    // Add neurons.
    if (sign > 0) {
        for (int i = 0; i < size; ++i) {
            // (ai; bi) = (0:02; 0:2) and (ci; di) = (-65; 8) + (15;-6)r2
            double a = 0.02;
            double b = 0.2; // increase for higher frequency oscillations

            double rand = fRand(0, 1);
            double c = -65.0 + (15.0 * rand * rand);

            rand = fRand(0, 1);
            double d = 8.0 - (6.0 * (rand * rand));
            this->add_neuron(a,b,c,d);
        }
    } else {
        for (int i = 0; i < size; ++i) {
            //(ai; bi) = (0:02; 0:25) + (0:08;-0:05)ri and (ci; di)=(-65; 2).
            double rand = fRand(0, 1);
            double a = 0.02 + (0.08 * rand);

            rand = fRand(0, 1);
            double b = 0.25 - (0.05 * rand);

            double c = -65.0;
            double d = 2.0;
            this->add_neuron(a,b,c,d);
        }
    }

    return layer_index;
}

// Adds a neuron to the environment.
// Returns the neuron's index.
int Environment::add_neuron(double a, double b, double c, double d) {
    this->neuron_parameters.push_back(NeuronParameters(a,b,c,d));
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
    cudaThreadSynchronize();
    cudaCheckError();
#endif
    this->update_voltages();
#ifdef parallel
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();
#endif
    this->timestep();
#ifdef parallel
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();
#endif
    this->update_weights();
#ifdef parallel
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();
#endif
}

/*
 * Performs activation during a timestep.
 * For each connectivity matrix, calculate sum of inputs for each neuron
 *   and add it to the current vector.
 */
void Environment::activate() {
    int* spikes = (int*)this->nat[SPIKE];
    double* currents = (double*)this->nat[CURRENT];

#ifdef parallel
    int threads = 64;
    int blocks = (this->num_neurons / threads) + (this->num_neurons % threads ? 1 : 0);
#endif

    /* 2. Activation */
    // For each connectivity matrix...
    //   Update Currents using synaptic input
    //     current += sign * dot ( spikes * weights )
    for (int cid = 0 ; cid < this->num_connections; ++cid) {
        ConnectivityMatrix &conn = this->connections[cid];
#ifdef parallel
        mult<<<blocks, threads>>>(
#else
        mult(
#endif
            conn.sign,
            spikes + conn.from_index,
            conn.matrix.mData,
            currents + conn.to_index,
            conn.from_size,
            conn.to_size);
    }
}


/*
 * Perform voltage update according to input currents using Izhikevich
 *   with Euler's method.
 */
void Environment::update_voltages() {
    /* 3. Voltage Updates */
#ifdef parallel
    int threads = 64;
    int blocks = (this->num_neurons / threads) + (this->num_neurons % threads ? 1 : 0);
    izhikevich<<<blocks, threads>>>(
#else
    izhikevich(
#endif
        (double*)this->nat[VOLTAGE],
        (double*)this->nat[RECOVERY],
        (double*)this->nat[CURRENT],
        (NeuronParameters*)this->nat[PARAMS],
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
    int threads = 64;
    int blocks = (this->num_neurons / threads) + (this->num_neurons % threads ? 1 : 0);
    calc_spikes<<<blocks, threads>>>(
#else
    calc_spikes(
#endif
        (int*)this->nat[SPIKE],
        (int*)this->nat[AGE],
        (double*)this->nat[VOLTAGE],
        (double*)this->nat[RECOVERY],
        (NeuronParameters*)this->nat[PARAMS],
        this->num_neurons);
}

/**
 * Updates weights.
 * TODO: implement.
 */
void Environment::update_weights() {
    /* 5. Update weights */
}
