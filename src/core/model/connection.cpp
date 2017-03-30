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
    this->init_params = config.init_params;
    this->row_field_size = 0;
    this->column_field_size = 0;
    this->row_stride = 0;
    this->column_stride = 0;
    this->row_offset = 0;
    this->column_offset = 0;

    switch (type) {
        case(FULLY_CONNECTED):
            this->num_weights = from_layer->size * to_layer->size;
            break;
        case(ONE_TO_ONE):
            if (from_layer->rows == to_layer->rows and from_layer->columns == to_layer->columns)
                this->num_weights = from_layer->size;
            else
                ErrorManager::get_instance()->log_error(
                    "Cannot connect differently sized layers one-to-one!");
            break;
        default:
            auto arborized_config = ArborizedConfig::decode(config.connection_params);
            this->row_field_size = arborized_config.row_field_size;
            this->column_field_size = arborized_config.column_field_size;
            this->row_stride = arborized_config.row_stride;
            this->column_stride = arborized_config.column_stride;
            this->row_offset = arborized_config.row_offset;
            this->column_offset = arborized_config.column_offset;

            // Because of checks in the kernels, mismatched layers will not cause
            //     problems.  Therefore, we only log a warning for this.
            if ((to_layer->rows != from_layer->rows and to_layer->rows !=
                    get_expected_rows(from_layer->rows, type, config.connection_params))
                or
                (to_layer->columns != from_layer->columns and to_layer->columns !=
                    get_expected_columns(from_layer->columns, type, config.connection_params)))
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

int get_expected_rows(int rows, ConnectionType type, std::string connection_params) {
    switch (type) {
        case(ONE_TO_ONE):
            return rows;
        case(FULLY_CONNECTED):
            return rows;
        default:
            auto arborized_config =
                ArborizedConfig::decode(connection_params);
            int row_field_size = arborized_config.row_field_size;
            int column_field_size = arborized_config.column_field_size;
            int row_stride = arborized_config.row_stride;
            int column_stride = arborized_config.column_stride;
            switch(type) {
                case(CONVERGENT):
                case(CONVOLUTIONAL):
                    return std::max(1,
                        1 + ((rows - row_field_size) / row_stride));
                case(DIVERGENT):
                    return std::max(1,
                        row_field_size + (row_stride * (rows - 1)));
                default:
                    ErrorManager::get_instance()->log_error(
                        "Invalid call to get_expected_rows!");
            }
    }
}

int get_expected_columns(int columns, ConnectionType type, std::string connection_params) {
    switch (type) {
        case(ONE_TO_ONE):
            return columns;
        case(FULLY_CONNECTED):
            return columns;
        default:
            auto arborized_config =
                ArborizedConfig::decode(connection_params);
            int row_field_size = arborized_config.row_field_size;
            int column_field_size = arborized_config.column_field_size;
            int row_stride = arborized_config.row_stride;
            int column_stride = arborized_config.column_stride;
            switch(type) {
                case(CONVERGENT):
                case(CONVOLUTIONAL):
                    return std::max(1,
                        1 + ((columns - column_field_size) / column_stride));
                case(DIVERGENT):
                    return std::max(1,
                        column_field_size + (column_stride * (columns - 1)));
                default:
                    ErrorManager::get_instance()->log_error(
                        "Invalid call to get_expected_columns!");
            }
    }
}
