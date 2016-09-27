#include "model.h"

Connection::Connection (int conn_id, Layer &from_layer, Layer &to_layer,
        bool plastic, int delay, float max_weight,
        ConnectionType type, OPCODE opcode) :
            id(conn_id),
            from_layer(from_layer),
            to_layer(to_layer),
            plastic(plastic),
            delay(delay),
            max_weight(max_weight),
            opcode(opcode),
            type(type),
            parent(-1) {
    if (delay >= (32 * HISTORY_SIZE))
        throw "Cannot implement connection delay longer than history!";

    if (type == FULLY_CONNECTED) {
        this->num_weights = from_layer.size * to_layer.size;
    } else if (type == ONE_TO_ONE) {
        if (from_layer.size != to_layer.size) {
            throw "Cannot connect differently sized layers one-to-one!";
        } else {
            this->num_weights = from_layer.size;
        }
    }
}

Connection::Connection(int conn_id,
        Layer &from_layer, Layer &to_layer, int parent) :
            id(conn_id),
            from_layer(from_layer),
            to_layer(to_layer),
            parent(parent) {
    if (type == FULLY_CONNECTED) {
        this->num_weights = from_layer.size * to_layer.size;
    } else if (type == ONE_TO_ONE) {
        this->num_weights = from_layer.size;
    }
}


Model::Model (std::string driver_string) :
        num_neurons(0),
        num_layers(0),
        num_connections(0),
        driver_string(driver_string) {}

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

int Model::add_neuron(std::string params) {
    this->parameter_strings.push_back(params);
    return this->num_neurons++;
}
