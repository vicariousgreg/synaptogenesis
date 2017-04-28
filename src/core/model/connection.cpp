#include "model/connection.h"
#include "model/layer.h"
#include "util/error_manager.h"

int Connection::count = 0;

Connection::Connection(Layer *from_layer, Layer *to_layer,
        ConnectionConfig *config) :
            id(Connection::count++),
            config(config),
            from_layer(from_layer),
            to_layer(to_layer),
            plastic(config->plastic),
            delay(config->delay),
            max_weight(config->max_weight),
            opcode(config->opcode),
            type(config->type),
            convolutional(type == CONVOLUTIONAL) {
    switch (type) {
        case(FULLY_CONNECTED): {
            auto fully_connected_config = config->get_fully_connected_config();
            if (fully_connected_config == nullptr) {
                config->set_fully_connected_config(
                    new FullyConnectedConfig(
                        0, from_layer->rows,
                        0, from_layer->columns,
                        0, to_layer->rows,
                        0, to_layer->columns));
                fully_connected_config = config->get_fully_connected_config();
            }
            if (not fully_connected_config->validate(this))
                ErrorManager::get_instance()->log_error(
                    "Invalid FullyConnectedConfig for connection!");
            this->num_weights = fully_connected_config->total_size;
            break;
        }
        case(ONE_TO_ONE):
            if (from_layer->rows == to_layer->rows and from_layer->columns == to_layer->columns)
                this->num_weights = from_layer->size;
            else
                ErrorManager::get_instance()->log_error(
                    "Cannot connect differently sized layers one-to-one!");
            break;
        default:
            auto arborized_config = config->get_arborized_config();
            if (arborized_config == nullptr)
                ErrorManager::get_instance()->log_error(
                    "Convergent/divergent connections require ArborizedConfig!");

            // Because of checks in the kernels, mismatched layers will not cause
            //     problems.  Therefore, we only log a warning for this.
            int expected_rows = config->get_expected_rows(from_layer->rows);
            int expected_columns = config->get_expected_columns(from_layer->columns);
            if ((to_layer->rows != from_layer->rows and to_layer->rows != expected_rows)
                or
                (to_layer->columns != from_layer->columns
                    and to_layer->columns != expected_columns))
                ErrorManager::get_instance()->log_warning(
                    "Unexpected destination layer size for arborized connection"
                    " from " + from_layer->name + " to " + to_layer->name +
                    " (" + std::to_string(to_layer->rows)
                        + ", " + std::to_string(to_layer->columns) + ") vs (" +
                    std::to_string(expected_rows) + ", "
                        + std::to_string(expected_columns) + ")!");

            switch (type) {
                case(CONVERGENT):
                    // Convergent connections use unshared mini weight matrices
                    // Each destination neuron connects to field_size squared neurons
                    this->num_weights =
                        arborized_config->row_field_size
                        * arborized_config->column_field_size
                        * to_layer->size;
                    break;
                case(CONVOLUTIONAL):
                    // Convolutional connections use a shared weight kernel
                    this->num_weights = arborized_config->get_total_field_size();
                    break;
                case(DIVERGENT):
                    // Divergent connections use unshared mini weight matrices
                    // Each source neuron connects to field_size squared neurons
                    this->num_weights =
                        arborized_config->get_total_field_size() * from_layer->size;
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
const ConnectionConfig* Connection::get_config() const { return config; }
