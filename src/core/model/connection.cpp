#include "model/connection.h"
#include "model/layer.h"
#include "util/error_manager.h"

Connection::Connection(Layer *from_layer, Layer *to_layer,
        ConnectionConfig *config) :
            config(config),
            from_layer(from_layer),
            to_layer(to_layer),
            plastic(config->plastic),
            delay(config->delay),
            max_weight(config->max_weight),
            opcode(config->opcode),
            type(config->type),
            convolutional(type == CONVOLUTIONAL) {
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
            this->row_field_size = config->arborized_config->row_field_size;
            this->column_field_size = config->arborized_config->column_field_size;
            this->row_stride = config->arborized_config->row_stride;
            this->column_stride = config->arborized_config->column_stride;
            this->row_offset = config->arborized_config->row_offset;
            this->column_offset = config->arborized_config->column_offset;

            // Because of checks in the kernels, mismatched layers will not cause
            //     problems.  Therefore, we only log a warning for this.
            if ((to_layer->rows != from_layer->rows and to_layer->rows !=
                    config->get_expected_rows(from_layer->rows)
                or
                (to_layer->columns != from_layer->columns and to_layer->columns !=
                    config->get_expected_columns(from_layer->columns))))
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

Connection::~Connection() {
    delete config;
}

int Connection::get_num_weights() const { return num_weights; }
int Connection::get_row_field_size() const { return row_field_size; }
int Connection::get_column_field_size() const { return column_field_size; }
int Connection::get_row_stride() const { return row_stride; }
int Connection::get_column_stride() const { return column_stride; }
int Connection::get_row_offset() const { return row_offset; }
int Connection::get_column_offset() const { return column_offset; }
const ConnectionConfig* Connection::get_config() const { return config; }
