#include "network/connection.h"
#include "network/layer.h"
#include "network/dendritic_node.h"
#include "network/structure.h"
#include "util/error_manager.h"

Connection::Connection(Layer *from_layer, Layer *to_layer,
        const ConnectionConfig *config) :
            config(config),
            from_layer(from_layer),
            to_layer(to_layer),
            node(to_layer->get_dendritic_node(config->dendrite, true)),
            plastic(config->plastic),
            delay(config->delay),
            max_weight(config->max_weight),
            opcode(config->opcode),
            type(config->type),
            convolutional(type == CONVOLUTIONAL),
            second_order(node->second_order),
            second_order_host(second_order and
                node->get_second_order_connection() == nullptr),
            second_order_slave(second_order and not second_order_host),
            name(config->name),
            id(std::hash<std::string>()(this->str())) {
    // Check for plastic second order connection
    if (second_order and plastic)
        LOG_ERROR(
            "Error in " + this->str() + ":\n"
            "  Plastic second order connections are not supported!");

    switch (type) {
        case FULLY_CONNECTED: {
            this->num_weights = from_layer->size * to_layer->size;
            break;
        }
        case SUBSET: {
            this->num_weights = config->get_subset_config().total_size;
            break;
        }
        case ONE_TO_ONE:
            if (from_layer->rows == to_layer->rows
                    and from_layer->columns == to_layer->columns)
                this->num_weights = to_layer->size;
            else
                LOG_ERROR(
                    "Error in " + this->str() + ":\n"
                    "  Cannot connect differently sized layers one-to-one!");
            break;
        default:
            auto arborized_config = config->get_arborized_config();

            switch (type) {
                case CONVOLUTIONAL:
                    // Convolutional connections use a shared weight kernel
                    this->num_weights = arborized_config.get_total_field_size();
                    break;
                case CONVERGENT:
                    // Convergent connections use unshared mini weight matrices
                    // Each destination neuron connects to field_size squared neurons
                    this->num_weights =
                        arborized_config.get_total_field_size() * to_layer->size;
                    break;
                case DIVERGENT:
                    // Divergent connections use unshared mini weight matrices
                    // Each source neuron connects to field_size squared neurons
                    this->num_weights =
                        arborized_config.get_total_field_size() * to_layer->size;

                    // Arithmetic operations for the divergent kernel constrain
                    //   the stride to non-zero values (division)
                    if (arborized_config.row_stride == 0 or
                        arborized_config.column_stride == 0)
                        LOG_ERROR(
                            "Error in " + this->str() + ":\n"
                            "  Divergent connections cannot have zero stride!");
                    break;
                default:
                    LOG_ERROR(
                        "Error in " + this->str() + ":\n"
                        "  Unknown layer connection type!");
            }
    }

    // If this is a non-host second order connection, match it to the weights
    //   of the host, not the size of the to_layer
    if (second_order_slave) {
        auto second_order_conn = node->get_second_order_connection();
        if (this->type != second_order_conn->type or
            this->num_weights != second_order_conn->get_num_weights())
            LOG_ERROR(
                "Error in " + this->str() + ":\n"
                "  Second order connection does not match host connection!");

        // Special case: Convolutional second order connections
        // Because these connections operate on the weights of the host
        //   connection, they are handled differently from first order
        //   convolutional connections.
        // Ensure that each convolution uses the same source set, since these
        //   connections are computed like one-to-one connections.
        auto arborized_config = config->get_arborized_config();
        if (this->type == CONVOLUTIONAL and
            (arborized_config.get_total_field_size() != from_layer->size
                or arborized_config.row_spacing != 1
                or arborized_config.column_spacing != 1
                or arborized_config.row_stride != 0
                or arborized_config.column_stride != 0))
            LOG_ERROR(
                "Error in " + this->str() + ":\n"
                "  Second order convolutional connections must have fields"
                " that are the size of the input layer, and must have 0 stride!");
    }

    // Validate the config
    if (not config->validate(this))
        LOG_ERROR(
            "Error in " + this->str() + ":\n"
            "  Invalid connection config!");

    // Assuming all went well, connect the layers
    from_layer->add_output_connection(this);
    to_layer->add_input_connection(this);
    node->add_child(this);
}

Connection::~Connection() {
    delete config;
}

std::string Connection::get_parameter(std::string key,
        std::string default_val) const {
    if (not this->get_config()->has(key))
        LOG_WARNING(
            "Error in " + this->str() + ":\n"
            "  Unspecified parameter: " + key
            + " -- using " + default_val + ".");
    return this->get_config()->get(key, default_val);
}

int Connection::get_num_weights() const { return num_weights; }
int Connection::get_compute_weights() const {
    if (type == CONVOLUTIONAL)
        if (second_order_host) return num_weights;
        else return num_weights * to_layer->size;
    else return num_weights;
}
const ConnectionConfig* Connection::get_config() const { return config; }

std::string Connection::str() const {
    return "[Connection: "
        + from_layer->name
        + " (" + from_layer->structure->name + ") -> "
        + to_layer->name
        + " (" + to_layer->structure->name + ") {Node: " + node->name + "}]";
}
