#include "model/structure.h"
#include "util/error_manager.h"

#define cimg_display 0
#include "CImg.h"

Structure::Structure(std::string name, ClusterType cluster_type)
        : name(name), cluster_type(cluster_type), total_neurons(0) { }

Structure::~Structure() {
    for (auto layer : layers) delete layer;
    for (auto conn : connections) delete conn;
}

Layer* Structure::find_layer(std::string name, bool log_error) {
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
        DendriticNode* node) {
    return to_structure->connect_layers(
        from_structure->find_layer(from_layer_name),
        to_structure->find_layer(to_layer_name),
        config, node);
}

Connection* Structure::connect_layers(
        Layer *from_layer, Layer *to_layer,
        ConnectionConfig *config,
        DendriticNode* node) {
    if (not config->validate())
        ErrorManager::get_instance()->log_error(
            "Error in " + this->str() + ":\n"
            "  Invalid connection config for connection from "
            + from_layer->str() + " to " + to_layer->str());

    if (node == nullptr) node = to_layer->dendritic_root;

    Connection *conn = new Connection(
        from_layer, to_layer, config, node);
    this->connections.push_back(conn);
    return conn;
}

Connection* Structure::connect_layers(
        std::string from_layer_name, std::string to_layer_name,
        ConnectionConfig *config,
        DendriticNode* node) {
    return connect_layers(
        find_layer(from_layer_name),
        find_layer(to_layer_name),
        config, node);
}

Connection* Structure::connect_layers_expected(
        std::string from_layer_name, LayerConfig *layer_config,
        ConnectionConfig *conn_config,
        DendriticNode* node) {
    Layer *from_layer = find_layer(from_layer_name);

    if (not conn_config->validate())
        ErrorManager::get_instance()->log_error(
            "Error in " + this->str() + ":\n"
            "  Invalid connection config for connection from "
            + from_layer->str() + " to " + layer_config->name);

    // Determine new layer size and create
    layer_config->rows =
        conn_config->get_expected_rows(from_layer->rows);
    layer_config->columns =
        conn_config->get_expected_columns(from_layer->columns);
    add_layer(layer_config);
    Layer *to_layer = find_layer(layer_config->name);

    // Connect new layer to given layer
    return connect_layers(from_layer, to_layer, conn_config, node);
}

Connection* Structure::connect_layers_matching(
        std::string from_layer_name,
        LayerConfig *layer_config, ConnectionConfig *conn_config,
        DendriticNode* node) {
    Layer *from_layer = find_layer(from_layer_name);

    // Determine new layer size and create
    layer_config->rows = from_layer->rows;
    layer_config->columns = from_layer->columns;
    add_layer(layer_config);
    Layer *to_layer = find_layer(layer_config->name);

    // Connect new layer to given layer
    return connect_layers(from_layer, to_layer, conn_config, node);
}

DendriticNode *Structure::get_dendritic_root(std::string to_layer_name) {
    return find_layer(to_layer_name)->dendritic_root;
}

DendriticNode *Structure::spawn_dendritic_node(std::string to_layer_name) {
    return find_layer(to_layer_name)->dendritic_root->add_child();
}

Layer* Structure::add_layer(LayerConfig *config) {
    if (find_layer(config->name, false) != nullptr)
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

void Structure::add_module(std::string layer_name, ModuleConfig *config) {
    find_layer(layer_name)->add_module(config);
}

void Structure::add_module_all(ModuleConfig *config) {
    for (auto layer : layers) layer->add_module(config);
}
