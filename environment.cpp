#include <cstdlib>
#include <ctime>
#include <iostream>

#include "environment.h"
#include "connectivity_matrix.h"
#include "tools.h"

#define SPIKE_THRESH 30
#define EULER_RES 10

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
    this->nat[PARAMS_A] = malloc(this->num_neurons * sizeof(double));
    this->nat[PARAMS_B] = malloc(this->num_neurons * sizeof(double));
    this->nat[PARAMS_C] = malloc(this->num_neurons * sizeof(double));
    this->nat[PARAMS_D] = malloc(this->num_neurons * sizeof(double));

    // Fill in table
    for (int i = 0 ; i < this->num_neurons ; ++i) {
        NeuronParameters params = this->neuron_parameters[i];
        ((double*)this->nat[PARAMS_A])[i] = params.a;
        ((double*)this->nat[PARAMS_B])[i] = params.b;
        ((double*)this->nat[PARAMS_C])[i] = params.c;
        ((double*)this->nat[PARAMS_D])[i] = params.d;

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
    //this->gather_input();
    this->activate();
    this->update_voltages();
    this->timestep();
    this->update_weights();
    this->report_output();
}

/*
 * Gathers input currents from external sources.
 * TODO: replace random generation with real input.
 */
void Environment::gather_input() {
    /* 1. Inputs */
    // Initialize currents to external currents
    for (int nid = 0 ; nid < this->num_neurons; ++nid) {
        ((double*)this->nat[CURRENT])[nid] = fRand(0, 5);
    }
}

/*
 * Performs activation during a timestep.
 * For each connectivity matrix, calculate sum of inputs for each neuron
 *   and add it to the current vector.
 */
void Environment::activate() {
    /* 2. Activation */
    // For each connectivity matrix...
    //   Update Currents using synaptic input
    //     current += sign * dot ( spikes * weights )
    for (int cid = 0 ; cid < this->num_connections; ++cid) {
        ConnectivityMatrix conn = this->connections[cid];
        for (int row = 0 ; row < conn.to_size ; ++row) {
            for (int col = 0 ; col < conn.from_size ; ++col) {
                ((double*)this->nat[CURRENT])[row + conn.to_index] += 
                    conn.sign *
                    ((int*)this->nat[SPIKE])[col + conn.from_index] *
                    conn.matrix(row, col);
            }
        }
    }
}

/*
 * Perform voltage update according to input currents using Izhikevich
 *   with Euler's method.
 */
void Environment::update_voltages() {
    /* 3. Voltage Updates */
    // For each neuron...
    //   update voltage according to current
    //     Euler's method with #defined resolution
    for (int nid = 0 ; nid < this->num_neurons; ++nid) {
        double voltage = ((double*)this->nat[VOLTAGE])[nid];
        double recovery = ((double*)this->nat[RECOVERY])[nid];
        double current = ((double*)this->nat[CURRENT])[nid];
        double delta_v = 0;

        double a = ((double*)this->nat[PARAMS_A])[nid];
        double b = ((double*)this->nat[PARAMS_B])[nid];
        double c = ((double*)this->nat[PARAMS_C])[nid];
        double d = ((double*)this->nat[PARAMS_D])[nid];

        // Euler's method for voltage/recovery update
        // If the voltage exceeds the spiking threshold, break
        for (int i = 0 ; i < EULER_RES && voltage < SPIKE_THRESH ; ++i) {
            delta_v = (0.04 * voltage * voltage) +
                            (5*voltage) + 140 - recovery + current;
            voltage += delta_v / EULER_RES;
            //recovery += a * ((b * voltage) - recovery)
            //                / EULER_RES;
        }
        recovery += a * ((b * voltage) - recovery);
        ((double*)this->nat[VOLTAGE])[nid] = voltage;
        ((double*)this->nat[RECOVERY])[nid] = recovery;
    }
}

/*
 * Perform timestep cycling.
 * Fills the spike buffer based on voltages and the SPIKE_THRESH.
 * Increments the ages of last spikes, and resets recovery if spiked.
 */
void Environment::timestep() {
    /* 4. Timestep */
    // Determine spikes.
    for (int i = 0; i < this->num_neurons; ++i) {
        int spike = ((double*)this->nat[VOLTAGE])[i] >= SPIKE_THRESH;
        ((int*)this->nat[SPIKE])[i] = spike;

        // Increment or reset spike ages.
        // Also, reset voltage if spiked.
        if (spike) {
            ((int*)this->nat[AGE])[i] = 0;
            ((double*)this->nat[VOLTAGE])[i] = ((double*)this->nat[PARAMS_C])[i];
            ((double*)this->nat[RECOVERY])[i] += ((double*)this->nat[PARAMS_D])[i];
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

/**
 * Reports spike vector.
 * TODO: implement.
 */
void Environment::report_output() {
    /* 6. Outputs */
}
