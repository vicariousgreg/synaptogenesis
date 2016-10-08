#include <sstream>

#include "model/model.h"
#include "io/input.h"
#include "io/output.h"

Connection::Connection (int conn_id, Layer *from_layer, Layer *to_layer,
        bool plastic, int delay, float max_weight,
        ConnectionType type, std::string params,  Opcode opcode) :
            id(conn_id),
            from_index(from_layer->index),
            from_size(from_layer->size),
            from_rows(from_layer->rows),
            from_columns(from_layer->columns),
            to_index(to_layer->index),
            to_size(to_layer->size),
            to_rows(to_layer->rows),
            to_columns(to_layer->columns),
            plastic(plastic),
            delay(delay),
            max_weight(max_weight),
            opcode(opcode),
            type(type),
            parent(-1) {
    if (delay >= (32 * HISTORY_SIZE))
        throw "Cannot implement connection delay longer than history!";

    switch (type) {
        case(FULLY_CONNECTED):
            this->params = params;
            this->num_weights = from_size * to_size;
            break;
        case(ONE_TO_ONE):
            this->params = params;
            if (from_rows == to_rows and from_columns == to_columns)
                this->num_weights = from_size;
            else throw "Cannot connect differently sized layers one-to-one!";
            break;
        default:
            std::stringstream stream(params);
            // Extract overlap
            if (stream.eof())
                throw "Overlap for arborized connection not specified!";
            stream >> this->overlap;
            if (this->overlap == 1)
                throw "Arborized connections cannot have overlap of 1!";

            // Extract stride
            if (stream.eof())
                throw "Stride for arborized connection not specified!";
            stream >> this->stride;

            // Extract remaining parameters for later
            if (!stream.eof()) std::getline(stream, this->params);

            if (to_rows != get_expected_dimension(from_rows, type, params) or
                to_columns != get_expected_dimension(from_columns, type, params))
                throw "Unexpected destination layer size for arborized connection!";
            this->num_weights = overlap * overlap * stride;

            switch (type) {
                case(DIVERGENT):
                    // Divergent connections use unshared mini weight matrices
                    // Each source neuron connects to overlap squared neurons
                    this->num_weights = overlap * overlap * from_size;
                    break;
                case(CONVERGENT):
                    // Convergent connections use unshared mini weight matrices
                    // Each destination neuron connects to overlap squared neurons
                    this->num_weights = overlap * overlap * to_size;
                    break;
                case(DIVERGENT_CONVOLUTIONAL):
                case(CONVERGENT_CONVOLUTIONAL):
                    this->convolutional = true;
                    // Convolutional connections use a shared weight kernel
                    this->num_weights = overlap * overlap;
                    break;
                default:
                    throw "Unknown layer connection type!";
            }
    }
}

Connection::Connection(int conn_id, Layer *from_layer, Layer *to_layer,
        Connection *parent) :
            id(conn_id),
            from_index(from_layer->index),
            from_size(from_layer->size),
            from_rows(from_layer->rows),
            from_columns(from_layer->columns),
            to_index(to_layer->index),
            to_size(to_layer->size),
            to_rows(to_layer->rows),
            to_columns(to_layer->columns),
            num_weights(parent->num_weights),
            plastic(parent->plastic),
            delay(parent->delay),
            max_weight(parent->max_weight),
            opcode(parent->opcode),
            type(parent->type),
            convolutional(parent->convolutional),
            overlap(parent->overlap),
            stride(parent->stride),
            params(parent->params),
            parent(parent->id) { }


Model::Model (std::string driver_string) :
        num_neurons(0),
        driver_string(driver_string) {}

int get_expected_dimension(int source_val, ConnectionType type, std::string params) {
    int overlap, stride;
    std::stringstream stream(params);

    switch (type) {
        case(ONE_TO_ONE):
            return source_val;
        case(DIVERGENT):
        case(DIVERGENT_CONVOLUTIONAL):
            stream >> overlap;
            stream >> stride;
            return overlap + (stride * (source_val -1));
        case(CONVERGENT):
        case(CONVERGENT_CONVOLUTIONAL):
            stream >> overlap;
            stream >> stride;
            return 1 + ((source_val - overlap) / stride);
        case(FULLY_CONNECTED):
        default:
            throw "Invalid call to get_expected_dimension!";
    }
}

int Model::connect_layers(int from_id, int to_id, bool plastic,
        int delay, float max_weight, ConnectionType type, Opcode opcode,
        std::string params) {
    int conn_id = this->connections.size();
    Connection *conn = new Connection(
        conn_id,
        this->layers[from_id], this->layers[to_id],
        plastic, delay, max_weight, type, params, opcode);
    this->connections.push_back(conn);
    return conn_id;
}

int Model::connect_layers_shared(int from_id, int to_id, int parent_id) {
    // Ensure parent doesn't have a parent
    if (this->connections[parent_id]->parent != -1)
        throw "Shared connections must refer to non-shared connection!";

    // Ensure that the weights can be shared by checking sizes
    Connection *parent = this->connections[parent_id];
    Layer *from_layer = this->layers[from_id];
    Layer *to_layer = this->layers[to_id];
    int conn_id = this->connections.size();

    if (from_layer->rows == parent->from_rows
            and from_layer->columns == parent->from_columns
            and to_layer->rows == parent->to_rows
            and to_layer->columns == parent->to_columns) {
        Connection *conn = new Connection(
            conn_id, from_layer, to_layer,
            this->connections[parent_id]);
        this->connections.push_back(conn);
        return conn_id;
    } else {
        throw "Cannot share weights between connections of different sizes!";
    }
}

int Model::connect_layers_expected(int from_id,
        std::string new_layer_params,bool plastic, int delay,
        float max_weight, ConnectionType type, Opcode opcode,
        std::string params) {
    Layer *from_layer = this->layers[from_id];

    // Determine new layer size and create
    int expected_rows = get_expected_dimension(
        from_layer->rows, type, params);
    int expected_columns = get_expected_dimension(
        from_layer->columns, type, params);
    int new_id = add_layer(
        expected_rows, expected_columns, new_layer_params);
    Layer *to_layer = this->layers[new_id];

    // Connect new layer to given layer
    int conn_id = this->connections.size();
    Connection *conn = new Connection(
        conn_id, from_layer, to_layer,
        plastic, delay, max_weight, type, params, opcode);
    this->connections.push_back(conn);

    // Return id of new layer
    return new_id;
}

int Model::add_layer(int rows, int columns, std::string params) {
    // Index of first neuron for layer
    int start_index = this->num_neurons;
    int layer_index = this->layers.size();

    this->layers.push_back(new Layer(layer_index, start_index, rows, columns));

    // Add neurons.
    this->add_neurons(rows*columns, params);

    return layer_index;
}

void Model::add_input(int layer, std::string type, std::string params) {
    Input *input = build_input(this->layers[layer], type, params);
    this->input_modules.push_back(input);
}

void Model::add_output(int layer, std::string type, std::string params) {
    Output *output = build_output(this->layers[layer], type, params);
    this->output_modules.push_back(output);
}
