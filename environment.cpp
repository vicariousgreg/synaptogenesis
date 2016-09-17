#include <cstdlib>
#include <ctime>
#include <iostream>

#include "environment.h"
#include "connectivity_matrix.h"
#include "tools.h"
#include "parallel.h"

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

    // Then initialize actual arrays
    this->nat[CURRENT] = malloc(this->num_neurons * sizeof(double));
    this->nat[VOLTAGE] = malloc(this->num_neurons * sizeof(double));
    this->nat[RECOVERY] = malloc(this->num_neurons * sizeof(double));
    this->nat[SPIKE] = malloc(this->num_neurons * sizeof(int));
    this->nat[AGE] = malloc(this->num_neurons * sizeof(int));
    this->nat[PARAMS] = malloc(this->num_neurons * sizeof(NeuronParameters));

    // Fill in table
    for (int i = 0 ; i < this->num_neurons ; ++i) {
        NeuronParameters params = this->neuron_parameters[i];
        ((NeuronParameters*)this->nat[PARAMS])[i] = this->neuron_parameters[i].copy();

        ((double*)this->nat[CURRENT])[i] = 0;
        ((double*)this->nat[VOLTAGE])[i] = params.c;
        ((double*)this->nat[RECOVERY])[i] = params.b * params.c;
        ((int*)this->nat[SPIKE])[i] = 0;
        ((int*)this->nat[AGE])[i] = 0;
    }
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

void Environment::set_random_currents(int layer_id, double max) {
    int offset = this->layers[layer_id].start_index;
    int size = this->layers[layer_id].size;

    for (int nid = offset ; nid < offset + size; ++nid) {
        ((double*)this->nat[CURRENT])[nid] = fRand(0, max);
    }
}

/******************************************************************************
 ************************ TIMESTEP DYNAMICS ***********************************
 ******************************************************************************/

/*
 * Performs a timestep cycle.
 */
void Environment::cycle() {
    this->activate();
    this->update_voltages();
    this->timestep();
    this->update_weights();
}

/*
 * Performs activation during a timestep.
 * For each connectivity matrix, calculate sum of inputs for each neuron
 *   and add it to the current vector.
 */
void Environment::activate() {
    int* spikes = (int*)this->nat[SPIKE];
    double* currents = (double*)this->nat[CURRENT];

    /* 2. Activation */
    // For each connectivity matrix...
    //   Update Currents using synaptic input
    //     current += sign * dot ( spikes * weights )
    for (int cid = 0 ; cid < this->num_connections; ++cid) {
        ConnectivityMatrix &conn = this->connections[cid];
        mult(
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
    izhikevich(
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
    int spike;
    NeuronParameters *params;

    /* 4. Timestep */
    // Determine spikes.
    for (int i = 0; i < this->num_neurons; ++i) {
        spike = ((double*)this->nat[VOLTAGE])[i] >= SPIKE_THRESH;
        ((int*)this->nat[SPIKE])[i] = spike;

        // Increment or reset spike ages.
        // Also, reset voltage if spiked.
        if (spike) {
            params = &(((NeuronParameters*)this->nat[PARAMS])[i]);
            ((int*)this->nat[AGE])[i] = 0;
            ((double*)this->nat[VOLTAGE])[i] = params->c;
            ((double*)this->nat[RECOVERY])[i] += params->d;
        } else {
            ((int*)this->nat[AGE])[i]++;
        }
    }
}

/**
 * Updates weights.
 * TODO: implement.
 */
void Environment::update_weights() {
    /* 5. Update weights */
}
