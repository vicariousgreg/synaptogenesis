#include "network/connection.h"
#include "network/layer.h"
#include "network/dendritic_node.h"
#include "network/structure.h"
#include "util/logger.h"

static int compute_num_weights(Layer *from_layer, Layer *to_layer,
        const ConnectionConfig *config) {
    // Compute this for error reporting
    auto node = to_layer->get_dendritic_node(config->dendrite, true);
    std::string str = "[Connection: "
        + from_layer->name
        + " (" + from_layer->structure->name + ") -> "
        + to_layer->name
        + " (" + to_layer->structure->name + ") {Node: " + node->name + "}]";

    switch (config->type) {
        case FULLY_CONNECTED:
            return from_layer->size * to_layer->size;
            break;
        case SUBSET:
            return config->get_subset_config().total_size;
            break;
        case ONE_TO_ONE:
            if (from_layer->rows == to_layer->rows
                    and from_layer->columns == to_layer->columns)
                return to_layer->size;
            else
                LOG_ERROR(
                    "Error in " + str + ":\n"
                    "  Cannot connect differently sized layers one-to-one!");
            break;
        case CONVERGENT:
        case DIVERGENT: {
            auto arborized_config = config->get_arborized_config();

            // Arithmetic operations for the divergent kernel constrain
            //   the stride to non-zero values (division)
            if (config->type == DIVERGENT) {
                if (arborized_config.row_stride == 0 or
                    arborized_config.column_stride == 0)
                    LOG_ERROR(
                        "Error in " + str + ":\n"
                        "  Divergent connections cannot have zero stride!");
            }

            // Convolutional connections use a shared weight kernel
            if (config->convolutional)
                return arborized_config.get_total_field_size();
            // Convergent connections use unshared mini weight matrices
            // Each destination neuron connects to field_size^2 neurons
            else
                return to_layer->size
                    * arborized_config.get_total_field_size();
        }
        default:
            LOG_ERROR(
                "Error in " + str + ":\n"
                "  Unknown layer connection type!");
    }
}

/* Copy constructor is meant for synapse_data, which can be sent to devices.
 * This will render pointers invalid, so they are set to nullptr when copied.
 * In addition, this ensures there are no double frees of pointers. */
Connection::Connection(const Connection& other)
    : config(nullptr),
      from_layer(nullptr),
      to_layer(nullptr),
      node(other.node),
      plastic(other.plastic),
      delay(other.delay),
      max_weight(other.max_weight),
      opcode(other.opcode),
      type(other.type),
      sparse(other.sparse),
      randomized_projection(other.randomized_projection),
      convolutional(other.convolutional),
      second_order(other.second_order),
      second_order_host(other.second_order_host),
      second_order_slave(other.second_order_slave),
      name(other.name),
      num_weights(other.num_weights),
      id(other.id) { }

Connection::Connection(Connection* other, ConnectionType new_type)
    : config(new ConnectionConfig(other->config)),
      from_layer(other->to_layer), // opposite from/to
      to_layer(other->from_layer),
      node(other->node),
      plastic(other->plastic),
      delay(other->delay),
      max_weight(other->max_weight),
      opcode(other->opcode),
      type(new_type),
      sparse(other->sparse),
      randomized_projection(other->randomized_projection),
      convolutional(other->convolutional),
      second_order(other->second_order),
      second_order_host(other->second_order_host),
      second_order_slave(other->second_order_slave),
      name(other->name),
      num_weights(other->num_weights),
      id(other->id) { }

/* Constructs an inverted arborized connection.
 * For use in randomizing divergent projections. */
Connection *Connection::invert(Connection* other) {
    switch (other->config->type) {
        case CONVERGENT:
            return new Connection(other, DIVERGENT);
            break;
        case DIVERGENT:
            return new Connection(other, CONVERGENT);
            break;
        default:
            LOG_ERROR("Connection invert method only "
                "intended for arborized connections!");
            break;
    }
    return nullptr;
}


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
            sparse(config->sparse or config->randomized_projection),
            randomized_projection(config->randomized_projection),
            convolutional(config->convolutional),
            second_order(node->second_order),
            second_order_host(second_order and
                node->get_second_order_connection() == nullptr),
            second_order_slave(second_order and not second_order_host),
            name(config->name),
            num_weights(compute_num_weights(from_layer, to_layer, config)),
            id(std::hash<std::string>()(this->str())) {
    // Check for plastic second order connection
    if (second_order and plastic)
        LOG_ERROR(
            "Error in " + this->str() + ":\n"
            "  Plastic second order connections are not supported!");

    // Check for sparse second order connection
    if (second_order and sparse)
        LOG_ERROR(
            "Error in " + this->str() + ":\n"
            "  Sparse second order connections are not supported!");

    // If this is a non-host second order connection, match it to the weights
    //   of the host, not the size of the to_layer
    if (second_order_slave) {
        auto second_order_conn = node->get_second_order_connection();
        if (this->type != second_order_conn->type or
            this->convolutional != second_order_conn->convolutional or
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
        if (this->convolutional and
            (arborized_config.get_total_field_size() != from_layer->size
                or arborized_config.row_spacing != 1
                or arborized_config.column_spacing != 1
                or arborized_config.row_stride != 0
                or arborized_config.column_stride != 0))
            LOG_ERROR(
                "Error in " + this->str() + ":\n"
                "  Second order convolutional connections must have fields"
                " that are the size of the input layer, and must have 0 stride"
                " and 1 spacing!");
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
    if (config != nullptr)
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

void Connection::sparsify(int sparse_num_weights) {
    this->num_weights = sparse_num_weights;
    this->type = SPARSE;
}

int Connection::get_compute_weights() const {
    if (convolutional and not second_order_slave)
        return num_weights * to_layer->size;
    else return num_weights;
}
const ConnectionConfig* Connection::get_config() const { return config; }

std::string Connection::str() const {
    return "[Connection: "
        + from_layer->name
        + " (" + from_layer->structure->name + ") -> "
        + to_layer->name
        + " (" + to_layer->structure->name + ") "
        + "{Node: " + node->name + "} name="
	+ this->name + "]";
}
