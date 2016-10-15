#include <sstream>

#include "connection.h"

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
            parent(NULL) {
    if (delay >= (32 * HISTORY_SIZE))
        throw "Cannot implement connection delay longer than history!";

    switch (type) {
        case(FULLY_CONNECTED):
            this->init_params = params;
            this->num_weights = from_size * to_size;
            break;
        case(ONE_TO_ONE):
            this->init_params = params;
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
            if (!stream.eof()) std::getline(stream, this->init_params);

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
            init_params(parent->init_params),
            parent(parent) { }

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
