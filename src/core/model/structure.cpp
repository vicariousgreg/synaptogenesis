#include "model/structure.h"
#include "util/error_manager.h"

#define cimg_display 0
#include "CImg.h"

Structure::Structure(std::string name, ClusterType cluster_type)
        : name(name), cluster_type(cluster_type) { }

Structure::~Structure() {
    for (auto layer : layers) delete layer;
    for (auto conn : connections) delete conn;
}

Layer* Structure::find_layer(std::string name, bool log_error) {
    try {
        return layers_by_name.at(name);
    } catch (...) {
        if (log_error)
            ErrorManager::get_instance()->log_error(
                "Could not find layer \"" + name + "\"!");
        else return nullptr;
    }
}

Connection* Structure::connect(
        Structure *from_structure, std::string from_layer_name,
        Structure *to_structure, std::string to_layer_name,
        ConnectionConfig config) {
    return to_structure->connect_layers(
        from_structure->find_layer(from_layer_name),
        to_structure->find_layer(to_layer_name),
        config);
}

Connection* Structure::connect_layers(
        Layer *from_layer, Layer *to_layer,
        ConnectionConfig config) {
    Connection *conn = new Connection(
        from_layer, to_layer, config);
    to_layer->add_to_root(conn);
    this->connections.push_back(conn);
    return conn;
}

Connection* Structure::connect_layers(
        std::string from_layer_name, std::string to_layer_name,
        ConnectionConfig config) {
    return connect_layers(
        find_layer(from_layer_name),
        find_layer(to_layer_name),
        config);
}

Connection* Structure::connect_layers_expected(
        std::string from_layer_name, LayerConfig layer_config,
        ConnectionConfig conn_config) {
    Layer *from_layer = find_layer(from_layer_name);

    // Determine new layer size and create
    layer_config.rows =
        conn_config.get_expected_rows(from_layer->rows);
    layer_config.columns =
        conn_config.get_expected_columns(from_layer->columns);
    add_layer(layer_config);
    Layer *to_layer = find_layer(layer_config.name);

    // Connect new layer to given layer
    Connection *conn = connect_layers(
        from_layer, to_layer, conn_config);
    return conn;
}

Connection* Structure::connect_layers_matching(
        std::string from_layer_name,
        LayerConfig layer_config, ConnectionConfig conn_config) {
    Layer *from_layer = find_layer(from_layer_name);

    // Determine new layer size and create
    layer_config.rows = from_layer->rows;
    layer_config.columns = from_layer->columns;
    add_layer(layer_config);
    Layer *to_layer = find_layer(layer_config.name);

    // Connect new layer to given layer
    Connection *conn = connect_layers(
        from_layer, to_layer, conn_config);
    return conn;
}

DendriticNode *Structure::get_dendritic_root(std::string to_layer_name) {
    return find_layer(to_layer_name)->dendritic_root;
}

DendriticNode *Structure::spawn_dendritic_node(std::string to_layer_name) {
    return find_layer(to_layer_name)->dendritic_root->add_child();
}

Connection* Structure::connect_layers_internal(
        DendriticNode *node, std::string from_layer_name,
        ConnectionConfig config) {
    Layer *from_layer = find_layer(from_layer_name);
    Layer *to_layer = node->to_layer;

    Connection *conn = new Connection(from_layer, to_layer, config);
    node->add_child(conn);
    this->connections.push_back(conn);
    return conn;
}

void Structure::add_layer(LayerConfig config) {
    if (find_layer(config.name, false) != nullptr)
        ErrorManager::get_instance()->log_error(
            "Repeated layer name!");

    Layer* layer = new Layer(this, config);
    this->layers.push_back(layer);
    this->layers_by_name[config.name] = layer;
    this->total_neurons += layer->size;
    this->neural_model_flags[config.neural_model] = true;
}

void Structure::add_layer_from_image(std::string path, LayerConfig config) {
    cimg_library::CImg<unsigned char> img(path.c_str());
    config.rows = img.height();
    config.columns = img.width();
    this->add_layer(config);
}

void Structure::add_module(std::string layer_name,
        std::string type, std::string params) {
    Layer *layer = find_layer(layer_name);
    layer->add_module(build_module(layer, type, params));
}
