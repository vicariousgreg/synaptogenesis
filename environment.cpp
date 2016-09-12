#include <cstdlib>
#include <iostream>
#include <ctime>

#include "environment.h"
#include "connectivity_matrix.h"
#include "tools.h"

#define SPIKE_THRESH 30
#define EULER_RES 10

int Environment::connect_layers(int from_layer, int to_layer, bool plastic) {
    this->connections.push_back(ConnectivityMatrix(
        this->layers[from_layer], this->layers[to_layer], plastic));
    return this->num_connections++;
}

/* Adds a layer to the environment.
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

// Adds a neuron to the environment.
// Returns the neuron's index.
int Environment::add_neuron(double a, double b, double c, double d) {
    this->ages.push_back(0);
    this->spikes.push_back(false);
    this->voltages.push_back(c);
    this->currents.push_back(0);
    this->recoveries.push_back(b * c);
    this->neuron_parameters.push_back(NeuronParameters(a,b,c,d));
    return this->num_neurons++;
}

/*
 * Performs a timestep cycle.
 */
void Environment::cycle() {
    /* 1. Inputs */
    // Initialize currents to external currents
    for (int nid = 0 ; nid < this->num_neurons; ++nid) {
        this->currents[nid] = fRand(0, 1);
    }

    /* 2. Activation */
    // For each connectivity matrix...
    //   Update Currents using synaptic input
    //     current += sign * dot ( spikes * weights )
    for (int cid = 0 ; cid < this->num_connections; ++cid) {
        ConnectivityMatrix conn = this->connections[cid];
        for (int row = 0 ; row < conn.to_size ; ++row) {
            for (int col = 0 ; col < conn.from_size ; ++col) {
                this->currents[row + conn.to_index] += 
                    conn.sign *
                    this->spikes[col + conn.from_index] *
                    conn.matrix(row, col);
            }
        }
    }

    // For each neuron...
    //   update voltage according to current
    //     Euler's method with #defined resolution
    for (int nid = 0 ; nid < this->num_neurons; ++nid) {
        double voltage = this->voltages[nid];
        double recovery = this->recoveries[nid];
        double current = this->currents[nid];
        double delta_v = 0;
        NeuronParameters params = this->neuron_parameters[nid];

        // Euler's method for voltage/recovery update
        // If the voltage exceeds the spiking threshold, break
        for (int i = 0 ; i < EULER_RES && voltage >= SPIKE_THRESH ; ++i) {
            delta_v = (0.04 * voltage * voltage) +
                            (5*voltage) + 140 - recovery + current;
            voltage += delta_v / EULER_RES;
            recovery += params.a * ((params.b * voltage) - recovery)
                            / EULER_RES;
        }
        //recovery += params.a * ((params.b * voltage) - recovery);
        this->voltages[nid] = voltage;
        this->recoveries[nid] = recovery;
    }

    /* 3. Timestep */
    // Determine spikes.
    for (int i = 0; i < this->num_neurons; ++i) {
        bool spike = this->voltages[i] >= SPIKE_THRESH;
        this->spikes[i] = spike;

        // Increment or reset spike ages.
        // Also, reset voltage if spiked.
        if (spike) {
            this->ages[i] = 0;
            this->voltages[i] = this->neuron_parameters[i].c;
            this->recoveries[i] += this->neuron_parameters[i].d;
        } else {
            ++this->ages[i];
        }
    }

    // Update weights.

    /* 4. Outputs */
}
