#include <sstream>

#include "model/connection.h"
#include "model/layer.h"
#include "util/error_manager.h"

Connection::Connection(Layer *from_layer, Layer *to_layer,
        ConnectionConfig config) :
            from_layer(from_layer),
            to_layer(to_layer),
            plastic(config.plastic),
            delay(config.delay),
            max_weight(config.max_weight),
            opcode(config.opcode),
            type(config.type),
            convolutional(type == CONVOLUTIONAL) {
    switch (type) {
        case(FULLY_CONNECTED):
            this->init_params = config.params;
            this->num_weights = from_layer->size * to_layer->size;
            break;
        case(ONE_TO_ONE):
            this->init_params = config.params;
            if (from_layer->rows == to_layer->rows and from_layer->columns == to_layer->columns)
                this->num_weights = from_layer->size;
            else
                ErrorManager::get_instance()->log_error(
                    "Cannot connect differently sized layers one-to-one!");
            break;
        default:
            std::stringstream stream(config.params);

            // Extract field size
            if (stream.eof())
                ErrorManager::get_instance()->log_error(
                    "Overlap for arborized connection not specified!");
            stream >> this->field_size;
            if (this->field_size == 1)
                ErrorManager::get_instance()->log_error(
                    "Arborized connections cannot have field size of 1!");

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
                    (this->stride == 1 and to_layer->rows == from_layer->rows
                    and to_layer->columns == from_layer->columns)
                and
                    (to_layer->rows !=
                        get_expected_dimension(from_layer->rows, type, config.params)
                    or to_layer->columns !=
                        get_expected_dimension(from_layer->columns, type, config.params)))
                ErrorManager::get_instance()->log_error(
                    "Unexpected destination layer size for arborized connection!");

            switch (type) {
                case(CONVERGENT):
                    // Convergent connections use unshared mini weight matrices
                    // Each destination neuron connects to field_size squared neurons
                    this->num_weights = field_size * field_size * to_layer->size;
                    break;
                case(CONVOLUTIONAL):
                    // Convolutional connections use a shared weight kernel
                    this->num_weights = field_size * field_size;
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

int get_expected_dimension(int source_val, ConnectionType type, std::string params) {
    int field_size, stride;
    std::stringstream stream(params);

    switch (type) {
        case(ONE_TO_ONE):
            return source_val;
        case(CONVERGENT):
        case(CONVOLUTIONAL):
            stream >> field_size;
            stream >> stride;
            return 1 + ((source_val - field_size) / stride);
        case(FULLY_CONNECTED):
            return source_val;
        default:
            ErrorManager::get_instance()->log_error(
                "Invalid call to get_expected_dimension!");
    }
}
