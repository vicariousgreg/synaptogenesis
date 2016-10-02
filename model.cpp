#include <sstream>

#include "model.h"
#include "input.h"
#include "output.h"

Connection::Connection (int conn_id, Layer *from_layer, Layer *to_layer,
        bool plastic, int delay, float max_weight,
        ConnectionType type, std::string params,  Opcode opcode) :
            id(conn_id),
            from_layer(from_layer),
            to_layer(to_layer),
            plastic(plastic),
            delay(delay),
            max_weight(max_weight),
            opcode(opcode),
            type(type),
            params(params),
            parent(-1) {
    if (delay >= (32 * HISTORY_SIZE))
        throw "Cannot implement connection delay longer than history!";

    std::stringstream stream(params);
    switch (type) {
        case(FULLY_CONNECTED):
            this->num_weights = from_layer->size * to_layer->size;
            break;
        case(ONE_TO_ONE):
            if (from_layer->matches_size(to_layer))
                this->num_weights = from_layer->size;
            else throw "Cannot connect differently sized layers one-to-one!";
            break;
        case(DIVERGENT):
            stream >> this->overlap;
            stream >> this->stride;
            if (to_layer->rows != get_expected_dimension(from_layer->rows, type, params) or
                to_layer->columns != get_expected_dimension(from_layer->columns, type, params))
                throw "Unexpected destination layer size for divergent connection!";
            this->num_weights = overlap * overlap * stride;
            // Divergent connections use unshared mini weight matrices
            // Each source neuron connects to overlap squared neurons
            this->num_weights = overlap * overlap * from_layer->size;
            break;
        case(CONVERGENT):
            stream >> this->overlap;
            stream >> this->stride;
            if (to_layer->rows != get_expected_dimension(from_layer->rows, type, params) or
                to_layer->columns != get_expected_dimension(from_layer->columns, type, params))
                throw "Unexpected destination layer size for convergent connection!";
            // Convergent connections use unshared mini weight matrices
            // Each destination neuron connects to overlap squared neurons
            this->num_weights = overlap * overlap * to_layer->size;
            break;
        case(CONVOLUTIONAL):
            stream >> this->overlap;
            stream >> this->stride;
            if (to_layer->rows != get_expected_dimension(from_layer->rows, type, params) or
                to_layer->columns != get_expected_dimension(from_layer->columns, type, params))
                throw "Unexpected destination layer size for convolutional connection!";
            // Convolutional connections use a shared weight kernel
            this->num_weights = overlap * overlap;
            break;
        default:
            throw "Unknown layer connection type!";
    }
}

Connection::Connection(int conn_id, Layer *from_layer, Layer *to_layer,
        Connection *parent) :
            id(conn_id),
            from_layer(from_layer),
            to_layer(to_layer),
            num_weights(parent->num_weights),
            plastic(parent->plastic),
            delay(parent->delay),
            max_weight(parent->max_weight),
            opcode(parent->opcode),
            type(parent->type),
            overlap(parent->overlap),
            stride(parent->stride),
            params(parent->params),
            parent(parent->id) { }


Model::Model (std::string driver_string) :
        num_neurons(0),
        num_layers(0),
        num_connections(0),
        driver_string(driver_string) {}

int Model::connect_layers(int from_layer, int to_layer, bool plastic,
        int delay, float max_weight, ConnectionType type, std::string params,
        Opcode opcode) {
    Connection *conn = new Connection(
        this->num_connections,
        this->layers[from_layer], this->layers[to_layer],
        plastic, delay, max_weight, type, params, opcode);
    this->connections.push_back(conn);
    return this->num_connections++;
}

int get_expected_dimension(int source_dimension, ConnectionType type, std::string params) {
    int overlap, stride;
    std::stringstream stream(params);

    switch (type) {
        case(ONE_TO_ONE):
            return source_dimension;
        case(DIVERGENT):
            stream >> overlap;
            stream >> stride;
            return overlap + (stride * (source_dimension -1));
        case(CONVERGENT):
        case(CONVOLUTIONAL):
            stream >> overlap;
            stream >> stride;
            return 1 + ((source_dimension - overlap) / stride);
        case(FULLY_CONNECTED):
        default:
            throw "Invalid call to get_expected_dimension!";
    }
}

int Model::connect_layers_shared(int from_layer, int to_layer, int parent_id) {
    // Ensure parent doesn't have a parent
    if (this->connections[parent_id]->parent != -1)
        throw "Shared connections must refer to non-shared connection!";

    // Ensure that the weights can be shared by checking sizes
    Connection *parent = this->connections[parent_id];
    if (this->layers[from_layer]->matches_size(parent->from_layer) and
            this->layers[to_layer]->matches_size(parent->to_layer)) {
        Connection *conn = new Connection(
            this->num_connections,
            this->layers[from_layer], this->layers[to_layer],
            this->connections[parent_id]);
        this->connections.push_back(conn);
        return this->num_connections++;
    } else {
        throw "Cannot share weights between connections of different sizes!";
    }
}

int Model::add_layer(int rows, int columns, std::string params) {
    // Index of first neuron for layer
    int start_index = this->num_neurons;
    int layer_index = this->num_layers++;

    this->layers.push_back(new Layer(layer_index, start_index, rows, columns));

    // Add neurons.
    this->add_neurons(rows*columns, params);

    return layer_index;
}

void Model::add_input(int layer, std::string type, std::string params) {
    Input *input = build_input(this->layers[layer], type, params);
    this->layers[layer]->input = input;
}

void Model::add_output(int layer, std::string type, std::string params) {
    Output *output = build_output(this->layers[layer], type, params);
    this->layers[layer]->output = output;
}
