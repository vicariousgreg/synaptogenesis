#include <cstdlib>

#include "model.h"

Model::Model () {
    this->num_neurons = 0;
    this->num_layers = 0;
    this->num_connections = 0;
}

int Model::connect_layers(int from_layer, int to_layer, bool plastic,
        int delay, float max_weight, ConnectionType type, OPCODE opcode) {
    Connection conn = Connection(
        this->num_connections,
        this->layers[from_layer], this->layers[to_layer],
        plastic, delay, max_weight, type, opcode);
    this->connections.push_back(conn);
    return this->num_connections++;
}

int Model::connect_layers_shared(int from_layer, int to_layer, int parent_id) {
    // Ensure parent doesn't have a parent
    if (this->connections[parent_id].parent != -1)
        throw "Shared connections must refer to non-shared connection!";

    // Ensure that the weights can be shared by checking sizes
    Connection parent = this->connections[parent_id];
    if (this->layers[from_layer].size != parent.from_layer.size or
            this->layers[to_layer].size != parent.to_layer.size) {
        throw "Cannot share weights between connections of different sizes!";
    }

    Connection conn = Connection(
        this->num_connections,
        this->layers[from_layer], this->layers[to_layer], parent_id);
    this->connections.push_back(conn);
    return this->num_connections++;
}

/*
 * Adds a layer to the environment.
 *     Adds the appropriate number of neurons according to the given size.
 *     Neurons are initialized with given parameters a,b,c,d.
 * Returns the layer's index.
 */
int Model::add_layer(int size, std::string params) {
    // Index of first neuron for layer
    int start_index = this->num_neurons;
    int layer_index = this->num_layers++;

    this->layers.push_back(Layer(layer_index, start_index, size));

    // Add neurons.
    for (int i = 0; i < size; ++i) {
        this->add_neuron(params);
    }

    return layer_index;
}

// Adds a neuron to the environment.
// Returns the neuron's index.
int Model::add_neuron(std::string params) {
    this->neuron_parameters.push_back(params);
    return this->num_neurons++;
}
