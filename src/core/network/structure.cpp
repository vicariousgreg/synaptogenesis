#include "network/structure.h"
#include "util/logger.h"

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

Layer* Structure::add_layer(const LayerConfig *layer_config) {
    this->config->add_layer(layer_config);
    return add_layer_internal(layer_config);
}

Layer* Structure::get_layer(std::string name, bool log_error) const {
    for (auto layer : layers)
        if (layer->name == name)
            return layer;

    if (log_error)
        LOG_ERROR(
            "Error in " + this->str() + ":\n"
            "  Could not find layer \"" + name + "\"");
    else return nullptr;
}

Connection* Structure::connect(
        Structure *from_structure,
        Structure *to_structure,
        const ConnectionConfig *conn_config) {
    auto from_layer = from_structure->get_layer(conn_config->from_layer);
    auto to_layer = to_structure->get_layer(conn_config->to_layer);

    Connection *conn = new Connection(
        from_layer, to_layer, conn_config);
    to_structure->connections.push_back(conn);
    return conn;
}

bool Structure::contains(std::string neural_model) const {
    return neural_model_flags.count(neural_model);
}

int Structure::get_num_neurons() const {
    int num_neurons = 0;
    for (auto layer : layers)
        num_neurons += layer->size;
    return num_neurons;
}

Layer* Structure::add_layer_internal(const LayerConfig *layer_config) {
    if (get_layer(layer_config->name, false) != nullptr)
        LOG_ERROR(
            "Error in " + this->str() + ":\n"
            "  Repeated layer name : \"" + layer_config->name + "\"");

    Layer* layer = new Layer(this, layer_config);
    this->layers.push_back(layer);
    this->neural_model_flags.insert(layer_config->neural_model);
    return layer;
}
