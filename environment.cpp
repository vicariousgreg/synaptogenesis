#include <iostream>
#include <vector>
#include "environment.h"

#define VOLTAGE_THRESHOLD 30
#define EULER_RESOLUTION 10

/* Adds a layer to the environment.
 *     Adds the appropriate number of neurons according to the given size.
 *     Neurons are initialized with given parameters a,b,c,d.
 *     |sign| indicates whether the layer is excitatory or inhibitory.
 * Returns the layer's index.
 * TODO: Add more parameters
 */
int Environment::add_layer(int size, int sign, double a, double b, double c, double d) {
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

    /* 2. Activation */
    // For each connectivity matrix...
    //   Update Currents using synaptic input
    //     current += sign * dot ( spikes * weights )
    //
    // For each neuron...
    //   update voltage according to current
    //     Euler's method with #defined resolution

    /* 3. Timestep */

    // Determine spikes.
    for (int i = 0; i < this->num_neurons; ++i) {
        bool spike = this->voltages[i] >= VOLTAGE_THRESHOLD;
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

int main(void) {
    Environment env;

    env.add_neuron(0,0,0,0);
    env.voltages[0] = 100;
    cout << env.voltages[0] << "\n";

    env.cycle();
    cout << env.voltages[0] << "\n";
    cout << env.spikes[0] << "\n";

    env.voltages[0] = 0;
    env.cycle();
    cout << env.spikes[0] << "\n";
    return 0;
}
