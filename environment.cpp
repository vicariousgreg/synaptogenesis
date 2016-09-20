#include <cstdlib>
#include <ctime>
#include <iostream>
#include <climits>
#include <stdio.h>
#include <math.h>

#include "environment.h"
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

/******************************************************************************
 ************************* GETTER / SETTER ************************************
 ******************************************************************************/

int* Environment::get_spikes() {
#ifdef parallel
    // Copy from GPU to local location
    cudaMemcpy(this->local_spikes, this->recent_spikes, this->num_neurons * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();
    return this->local_spikes;
#else
    return this->recent_spikes;
#endif
}

float* Environment::get_currents() {
#ifdef parallel
    // Copy from GPU to local location
    cudaMemcpy(this->local_currents, this->nat[CURRENT], this->num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();
    return this->local_currents;
#else
    return (float*)this->nat[CURRENT];
#endif
}

void Environment::inject_random_current(int layer_id, float max) {
    int offset = this->layers[layer_id].start_index;
    int size = this->layers[layer_id].size;
#ifdef parallel
    // Send to GPU
    float* temp = (float*)malloc(size * sizeof(float));
    for (int nid = 0 ; nid < size; ++nid) {
        temp[nid] = fRand(0, max);
    }
    this->inject_current(layer_id, temp);
    free(temp);
#else
    for (int nid = offset ; nid < offset + size; ++nid) {
        ((float*)this->nat[CURRENT])[nid] = fRand(0, max);
    }
#endif
}

void Environment::inject_current(int layer_id, float* input) {
    int offset = this->layers[layer_id].start_index;
    int size = this->layers[layer_id].size;
#ifdef parallel
    // Send to GPU
    void* current = &((float*)this->nat[CURRENT])[offset];
    cudaMemcpy(current, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();
#else
    for (int nid = 0 ; nid < size; ++nid) {
        ((float*)this->nat[CURRENT])[nid+offset] = input[nid];
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

    // Build weight matrices
    for (int i = 0 ; i < this->num_connections ; ++i)
        this->connections[i].build();

#ifdef parallel
    // Local spikes for output reporting
    this->local_spikes = (int*)calloc(count, sizeof(int));
    this->local_currents = (float*)calloc(count, sizeof(float));

    // Allocate space on GPU
    cudaMalloc(&this->nat[CURRENT], count * sizeof(float));
    cudaMalloc(&this->nat[VOLTAGE], count * sizeof(float));
    cudaMalloc(&this->nat[RECOVERY], count * sizeof(float));
    cudaMalloc(&this->nat[PARAMS], count * sizeof(NeuronParameters));

    // Set up spikes, keep pointer to recent spikes.
    cudaMalloc(&this->spikes, HISTORY_SIZE * count * sizeof(int));
    this->recent_spikes = &this->spikes[(HISTORY_SIZE-1) * count];

    // Make temporary arrays for initialization
    float *temp_current = (float*)malloc(count * sizeof(float));
    float *temp_voltage = (float*)malloc(count * sizeof(float));
    float *temp_recovery = (float*)malloc(count * sizeof(float));
    int *temp_spike = (int*)calloc(count, HISTORY_SIZE * sizeof(int));
    NeuronParameters *temp_params = (NeuronParameters*)malloc(count * sizeof(NeuronParameters));

    // Fill in table
    for (int i = 0 ; i < count ; ++i) {
        NeuronParameters &params = this->neuron_parameters[i];
        temp_params[i] = this->neuron_parameters[i].copy();

        temp_current[i] = 0;
        temp_voltage[i] = params.c;
        temp_recovery[i] = params.b * params.c;
    }

    // Copy values to GPU
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();

    cudaMemcpy(this->nat[CURRENT], temp_current, count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->nat[VOLTAGE], temp_voltage, count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->nat[RECOVERY], temp_recovery, count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->nat[PARAMS], temp_params, count * sizeof(NeuronParameters), cudaMemcpyHostToDevice);
    cudaMemcpy(this->spikes, temp_spike, count * HISTORY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    cudaCheckError();

    // Free up temporary memory
    free(temp_current);
    free(temp_voltage);
    free(temp_recovery);
    free(temp_spike);
    free(temp_params);

#else
    // Then initialize actual arrays
    this->nat[CURRENT] = malloc(count * sizeof(float));
    this->nat[VOLTAGE] = malloc(count * sizeof(float));
    this->nat[RECOVERY] = malloc(count * sizeof(float));
    this->nat[PARAMS] = malloc(count * sizeof(NeuronParameters));

    // Set up spikes array
    // Keep track of pointer to least significant word for convenience
    this->spikes = (int*)calloc(count, HISTORY_SIZE * sizeof(int));
    this->recent_spikes = &this->spikes[(HISTORY_SIZE-1) * count];

    // Fill in table
    for (int i = 0 ; i < count ; ++i) {
        NeuronParameters &params = this->neuron_parameters[i];
        ((NeuronParameters*)this->nat[PARAMS])[i] = this->neuron_parameters[i].copy();

        ((float*)this->nat[CURRENT])[i] = 0;
        ((float*)this->nat[VOLTAGE])[i] = params.c;
        ((float*)this->nat[RECOVERY])[i] = params.b * params.c;
    }
#endif
}

/*
 * Connects two layers with a weight matrix.
 * If the layer is plastic, it will learn.
 * TODO: this initializes with random weights. provide means of changing that
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
 * For each weight matrix, calculate sum of inputs for each neuron
 *   and add it to the current vector.
 */
void Environment::activate() {
    float* currents = (float*)this->nat[CURRENT];

    /* 2. Activation */
    // For each weight matrix...
    //   Update Currents using synaptic input
    //     current += sign * dot ( spikes * weights )
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
            this->recent_spikes + conn.from_index,  // only most recent
            conn.mData,
            currents + conn.to_index,
            conn.from_size,
            conn.to_size);
#ifdef parallel
        cudaDeviceSynchronize();
        cudaThreadSynchronize();
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
        (float*)this->nat[VOLTAGE],
        (float*)this->nat[RECOVERY],
        (float*)this->nat[CURRENT],
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
    int threads = 32;
    int blocks = ceil((float)(this->num_neurons) / threads);
    calc_spikes<<<blocks, threads>>>(
#else
    calc_spikes(
#endif
        this->spikes,
        (float*)this->nat[VOLTAGE],
        (float*)this->nat[RECOVERY],
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
