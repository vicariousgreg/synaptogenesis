#include "network/structure.h"
#include "util/error_manager.h"

#define cimg_display 0
#include "CImg.h"

Structure::Structure(StructureConfig* config)
        : name(config->name),
          cluster_type(config->cluster_type),
          config(config) {
    for (auto layer_config : config->get_layers())
        this->add_layer_internal(layer_config);
}

Structure::Structure(std::string name, ClusterType cluster_type)
        : name(name),
          cluster_type(cluster_type),
          config(new StructureConfig(name, cluster_type)) { }

Structure::~Structure() {
    for (auto layer : layers) delete layer;
    for (auto conn : connections) delete conn;
    delete config;
}

Layer* Structure::get_layer(std::string name, bool log_error) const {
    for (auto layer : layers)
        if (layer->name == name)
            return layer;

    if (log_error)
        ErrorManager::get_instance()->log_error(
            "Error in " + this->str() + ":\n"
            "  Could not find layer \"" + name + "\"");
    else return nullptr;
}

Connection* Structure::connect(
        Structure *from_structure, std::string from_layer_name,
        Structure *to_structure, std::string to_layer_name,
        ConnectionConfig *conn_config,
        std::string node, std::string name) {
    return to_structure->connect_layers(
        from_structure->get_layer(from_layer_name),
        to_structure->get_layer(to_layer_name),
        conn_config, node, name);
}

Connection* Structure::connect_layers(
        Layer *from_layer, Layer *to_layer,
        ConnectionConfig *conn_config,
        std::string node, std::string name) {
    Connection *conn = new Connection(
        from_layer, to_layer, conn_config,
        to_layer->get_dendritic_node(node),
        name);
    this->connections.push_back(conn);
    return conn;
}

std::string Structure::get_parent_node_name(Connection *conn) const {
    for (auto node : conn->to_layer->get_dendritic_nodes())
        if (node->is_leaf() and conn == node->conn)
            return node->parent->name;
        else if (node->second_order
                and conn == node->get_second_order_connection())
            return node->name;

    ErrorManager::get_instance()->log_error(
        "Error in " + this->str() + ":\n"
        "  Could not find parent node for connection: " + conn->str());
}

Layer* Structure::add_layer_internal(LayerConfig *layer_config) {
    if (get_layer(layer_config->name, false) != nullptr)
        ErrorManager::get_instance()->log_error(
            "Error in " + this->str() + ":\n"
            "  Repeated layer name : \"" + layer_config->name + "\"");

    Layer* layer = new Layer(this, layer_config);
    this->layers.push_back(layer);
    this->neural_model_flags.insert(layer_config->neural_model);
    return layer;
}

Layer* Structure::add_layer(LayerConfig *layer_config) {
    this->config->add_layer(layer_config);
    return add_layer_internal(layer_config);
}
