#include <sstream>

#include "model/connection.h"
#include "util/error_manager.h"

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
    switch (type) {
        case(FULLY_CONNECTED):
            this->init_params = params;
            this->num_weights = from_layer->size * to_layer->size;
            break;
        case(ONE_TO_ONE):
            this->init_params = params;
            if (from_layer->rows == to_layer->rows and from_layer->columns == to_layer->columns)
                this->num_weights = from_layer->size;
            else
                ErrorManager::get_instance()->log_error(
                    "Cannot connect differently sized layers one-to-one!");
            break;
        default:
            std::stringstream stream(params);
            // Extract overlap
            if (stream.eof())
                ErrorManager::get_instance()->log_error(
                    "Overlap for arborized connection not specified!");
            stream >> this->overlap;
            if (this->overlap == 1)
                ErrorManager::get_instance()->log_error(
                    "Arborized connections cannot have overlap of 1!");
            else if (this->overlap % 2 == 0)
                ErrorManager::get_instance()->log_error(
                    "Arborized connections cannot have an even overlap!");

            // Extract stride
            if (stream.eof())
                ErrorManager::get_instance()->log_error(
                    "Stride for arborized connection not specified!");
            stream >> this->stride;

            // Extract remaining parameters for later
            if (!stream.eof()) std::getline(stream, this->init_params);

            // If the layers are the same size, arborized connections can be
            //     accommodated.  If not, the layers must meet size expectations
            if (not
                    (to_layer->rows == from_layer->rows
                    and to_layer->columns == from_layer->columns)
                and
                    (to_layer->rows !=
                        get_expected_dimension(from_layer->rows, type, params)
                    or to_layer->columns !=
                        get_expected_dimension(from_layer->columns, type, params)))
                ErrorManager::get_instance()->log_error(
                    "Unexpected destination layer size for arborized connection!");
            this->num_weights = overlap * overlap * stride;

            switch (type) {
                case(CONVERGENT):
                    // Convergent connections use unshared mini weight matrices
                    // Each destination neuron connects to overlap squared neurons
                    this->num_weights = overlap * overlap * to_layer->size;
                    break;
                case(CONVOLUTIONAL):
                    this->convolutional = true;
                    // Convolutional connections use a shared weight kernel
                    this->num_weights = overlap * overlap;
                    break;
                default:
                    ErrorManager::get_instance()->log_error(
                        "Unknown layer connection type!");
            }
    }

    // Assuming all went well, connect the layers
    from_layer->add_output_connection(this);
    to_layer->add_input_connection(this);
}

Connection::Connection(int conn_id, Layer *from_layer, Layer *to_layer,
        Connection *parent) :
            Connection(conn_id, from_layer, to_layer,
                        parent->plastic, parent->delay,
                        parent->max_weight, parent->type,
                        parent->init_params, parent->opcode) {
    this->parent = parent;
}

int get_expected_dimension(int source_val, ConnectionType type, std::string params) {
    int overlap, stride;
    std::stringstream stream(params);

    switch (type) {
        case(ONE_TO_ONE):
            return source_val;
        case(CONVERGENT):
        case(CONVOLUTIONAL):
            stream >> overlap;
            stream >> stride;
            return 1 + ((source_val - overlap) / stride);
        case(FULLY_CONNECTED):
            return source_val;
        default:
            ErrorManager::get_instance()->log_error(
                "Invalid call to get_expected_dimension!");
    }
}
