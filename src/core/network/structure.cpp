#include "network/structure.h"
#include "util/error_manager.h"

#define cimg_display 0
#include "CImg.h"

Structure::Structure(std::string name, ClusterType cluster_type)
        : name(name), cluster_type(cluster_type), total_neurons(0) { }

Structure::~Structure() {
    for (auto layer : layers) delete layer;
    for (auto conn : connections) delete conn;
}

Layer* Structure::get_layer(std::string name, bool log_error) {
    try {
        return layers_by_name.at(name);
    } catch (std::out_of_range) {
        if (log_error)
            ErrorManager::get_instance()->log_error(
                "Error in " + this->str() + ":\n"
                "  Could not find layer \"" + name + "\"");
        else return nullptr;
    }
}

Connection* Structure::connect(
        Structure *from_structure, std::string from_layer_name,
        Structure *to_structure, std::string to_layer_name,
        ConnectionConfig *config,
        std::string node, std::string name) {
    return to_structure->connect_layers(
        from_structure->get_layer(from_layer_name),
        to_structure->get_layer(to_layer_name),
        config, node, name);
}

Connection* Structure::connect_layers(
        Layer *from_layer, Layer *to_layer,
        ConnectionConfig *config,
        std::string node, std::string name) {
    Connection *conn = new Connection(
        from_layer, to_layer, config,
        to_layer->get_dendritic_node(node),
        name);
    this->connections.push_back(conn);
    return conn;
}

Connection* Structure::connect_layers(
        std::string from_layer_name, std::string to_layer_name,
        ConnectionConfig *config,
        std::string node,
        std::string name) {
    return connect_layers(
        get_layer(from_layer_name),
        get_layer(to_layer_name),
        config, node, name);
}

Connection* Structure::connect_layers_expected(
        std::string from_layer_name, LayerConfig *layer_config,
        ConnectionConfig *conn_config,
        std::string node,
        std::string name) {
    Layer *from_layer = get_layer(from_layer_name);

    // Determine new layer size and create
    layer_config->rows =
        conn_config->get_expected_rows(from_layer->rows);
    layer_config->columns =
        conn_config->get_expected_columns(from_layer->columns);
    add_layer(layer_config);
    Layer *to_layer = get_layer(layer_config->name);

    // Connect new layer to given layer
    return connect_layers(from_layer, to_layer, conn_config, node, name);
}

Connection* Structure::connect_layers_matching(
        std::string from_layer_name,
        LayerConfig *layer_config, ConnectionConfig *conn_config,
        std::string node,
        std::string name) {
    Layer *from_layer = get_layer(from_layer_name);

    // Determine new layer size and create
    layer_config->rows = from_layer->rows;
    layer_config->columns = from_layer->columns;
    add_layer(layer_config);
    Layer *to_layer = get_layer(layer_config->name);

    // Connect new layer to given layer
    return connect_layers(from_layer, to_layer, conn_config, node, name);
}

void Structure::set_second_order(std::string to_layer_name,
        std::string node_name) {
    get_layer(to_layer_name)
        ->get_dendritic_node(node_name)
        ->set_second_order();
}
void Structure::create_dendritic_node(std::string to_layer_name,
        std::string parent_node_name, std::string child_node_name) {
    get_layer(to_layer_name)
        ->get_dendritic_node(parent_node_name)
        ->add_child(child_node_name);
}

std::string Structure::get_parent_node_name(Connection *conn) const {
    for (auto node : conn->to_layer->get_dendritic_nodes())
        if (node->is_leaf() and conn == node->conn)
            return node->parent->name;
        else if (node->is_second_order()
                and conn == node->get_second_order_connection())
            return node->name;

    ErrorManager::get_instance()->log_error(
        "Error in " + this->str() + ":\n"
        "  Could not find parent node for connection: " + conn->str());
}

Layer* Structure::add_layer(LayerConfig *config) {
    if (get_layer(config->name, false) != nullptr)
        ErrorManager::get_instance()->log_error(
            "Error in " + this->str() + ":\n"
            "  Repeated layer name : \"" + config->name + "\"");

    Layer* layer = new Layer(this, config);
    this->layers.push_back(layer);
    this->layers_by_name[config->name] = layer;
    this->total_neurons += layer->size;
    this->neural_model_flags.insert(config->neural_model);
    return layer;
}

Layer* Structure::add_layer_from_image(std::string path, LayerConfig *config) {
    cimg_library::CImg<unsigned char> img(path.c_str());
    config->rows = img.height();
    config->columns = img.width();
    return this->add_layer(config);
}
