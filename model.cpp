#include <cstdlib>
#include <stdio.h>

#include "model.h"
#include "tools.h"

Model::Model () {
    this->num_neurons = 0;
    this->num_layers = 0;
    this->num_connections = 0;
}

/*
 * Connects two layers with a weight matrix.
 * If the layer is plastic, it will learn.
 */
int Model::connect_layers(int from_layer, int to_layer,
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
int Model::add_layer(int size, int sign,
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
int Model::add_randomized_layer(
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
int Model::add_neuron(float a, float b, float c, float d) {
    this->neuron_parameters.push_back(NeuronParameters(a,b,c,d));
    return this->num_neurons++;
}
