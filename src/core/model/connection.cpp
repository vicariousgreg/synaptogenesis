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
                    "Row field size for arborized connection not specified!");
            stream >> this->row_field_size;
            if (this->row_field_size == 1)
                ErrorManager::get_instance()->log_error(
                    "Arborized connections cannot have field size of 1!");
            if (stream.eof())
                ErrorManager::get_instance()->log_error(
                    "Column field size for arborized connection not specified!");
            stream >> this->column_field_size;
            if (this->column_field_size == 1)
                ErrorManager::get_instance()->log_error(
                    "Arborized connections cannot have field size of 1!");

            // Extract stride
            if (stream.eof())
                ErrorManager::get_instance()->log_error(
                    "Row stride for arborized connection not specified!");
            stream >> this->row_stride;
            if (stream.eof())
                ErrorManager::get_instance()->log_error(
                    "Column stride for arborized connection not specified!");
            stream >> this->column_stride;

            // Extract remaining parameters for later
            if (!stream.eof()) std::getline(stream, this->init_params);

            // Because of checks in the kernels, mismatched layers will not cause
            //     problems.  Therefore, we only log a warning for this.
            if ((to_layer->rows != from_layer->rows and to_layer->rows !=
                    get_expected_rows(from_layer->rows, type, config.params))
                or
                (to_layer->columns != from_layer->columns and to_layer->columns !=
                    get_expected_columns(from_layer->columns, type, config.params)))
                ErrorManager::get_instance()->log_warning(
                    "Unexpected destination layer size for arborized connection!");

            switch (type) {
                case(CONVERGENT):
                    // Convergent connections use unshared mini weight matrices
                    // Each destination neuron connects to field_size squared neurons
                    this->num_weights = row_field_size * column_field_size * to_layer->size;
                    break;
                case(CONVOLUTIONAL):
                    // Convolutional connections use a shared weight kernel
                    this->num_weights = row_field_size * column_field_size;
                    break;
                case(DIVERGENT):
                    // Divergent connections use unshared mini weight matrices
                    // Each source neuron connects to field_size squared neurons
                    this->num_weights = row_field_size * column_field_size * from_layer->size;
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

int get_expected_rows(int rows, ConnectionType type, std::string params) {
    int row_field_size, row_stride;
    int col_field_size, col_stride;
    std::stringstream stream(params);

    switch (type) {
        case(ONE_TO_ONE):
            return rows;
        case(CONVERGENT):
        case(CONVOLUTIONAL):
            stream >> row_field_size;
            stream >> col_field_size;
            stream >> row_stride;
            stream >> col_stride;
            return 1 + ((rows - row_field_size) / row_stride);
        case(DIVERGENT):
            stream >> row_field_size;
            stream >> col_field_size;
            stream >> row_stride;
            stream >> col_stride;
            return row_field_size + (row_stride * (rows - 1));
        case(FULLY_CONNECTED):
            return rows;
        default:
            ErrorManager::get_instance()->log_error(
                "Invalid call to get_expected_rows!");
    }
}

int get_expected_columns(int columns, ConnectionType type, std::string params) {
    int row_field_size, row_stride;
    int column_field_size, column_stride;
    std::stringstream stream(params);

    switch (type) {
        case(ONE_TO_ONE):
            return columns;
        case(CONVERGENT):
        case(CONVOLUTIONAL):
            stream >> row_field_size;
            stream >> column_field_size;
            stream >> row_stride;
            stream >> column_stride;
            return 1 + ((columns - column_field_size) / column_stride);
        case(DIVERGENT):
            stream >> row_field_size;
            stream >> column_field_size;
            stream >> row_stride;
            stream >> column_stride;
            return column_field_size + (column_stride * (columns - 1));
        case(FULLY_CONNECTED):
            return columns;
        default:
            ErrorManager::get_instance()->log_error(
                "Invalid call to get_expected_columns!");
    }
}
