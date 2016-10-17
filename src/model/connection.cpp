#include <sstream>

#include "connection.h"

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
            parent(NULL) {
    if (delay >= (32 * HISTORY_SIZE))
        throw "Cannot implement connection delay longer than history!";

    switch (type) {
        case(FULLY_CONNECTED):
            this->init_params = params;
            this->num_weights = from_layer->size * to_layer->size;
            break;
        case(ONE_TO_ONE):
            this->init_params = params;
            if (from_layer->rows == to_layer->rows and from_layer->columns == to_layer->columns)
                this->num_weights = from_layer->size;
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

            if (to_layer->rows != get_expected_dimension(from_layer->rows, type, params) or
                to_layer->columns != get_expected_dimension(from_layer->columns, type, params))
                throw "Unexpected destination layer size for arborized connection!";
            this->num_weights = overlap * overlap * stride;

            switch (type) {
                case(DIVERGENT):
                    // Divergent connections use unshared mini weight matrices
                    // Each source neuron connects to overlap squared neurons
                    this->num_weights = overlap * overlap * from_layer->size;
                    break;
                case(CONVERGENT):
                    // Convergent connections use unshared mini weight matrices
                    // Each destination neuron connects to overlap squared neurons
                    this->num_weights = overlap * overlap * to_layer->size;
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
            from_layer(from_layer),
            to_layer(to_layer),
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
