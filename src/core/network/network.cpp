#include "network/network.h"
#include "util/error_manager.h"
#include "builder.h"

Network::Network(NetworkConfig* config) : config(config) {
    for (auto structure : config->get_structures())
        this->add_structure_internal(structure);
    for (auto conn : config->get_connections())
        this->add_connection_internal(conn);
}

Network::~Network() {
    for (auto& structure : structures) delete structure;
    delete config;
}

Network* Network::load(std::string path) {
    return load_network(path);
}

void Network::save(std::string path) {
    save_network(this, path);
}

void Network::add_structure(StructureConfig *struct_config) {
    this->add_structure_internal(struct_config);
    this->config->add_structure(struct_config);
}

Structure* Network::get_structure(std::string name, bool log_error) {
    Structure *structure = nullptr;
    for (auto s : this->structures)
        if (s->name == name)
            structure = s;
    if (structure == nullptr and log_error)
            LOG_ERROR(
                "Could not find structure: " + name);
    return structure;
}

void Network::add_connection(const ConnectionConfig* conn_config) {
    this->add_connection_internal(conn_config);
    this->config->add_connection(conn_config);
}

const LayerList Network::get_layers() const {
    LayerList layers;
    for (auto& structure : structures)
        for (auto& layer : structure->get_layers())
            layers.push_back(layer);
    return layers;
}

const LayerList Network::get_layers(std::string neural_model) const {
    LayerList layers;
    for (auto& structure : structures)
        for (auto& layer : structure->get_layers())
            if (layer->neural_model == neural_model)
                layers.push_back(layer);
    return layers;
}

const ConnectionList Network::get_connections() const {
    ConnectionList connections;
    for (auto& structure : structures)
        for (auto& conn : structure->get_connections())
            connections.push_back(conn);
    return connections;
}

Connection* Network::get_connection(std::string name, bool log_error) {
    Connection *conn = nullptr;
    for (auto c : this->get_connections())
        if (c->name == name)
            conn = c;
    if (conn == nullptr and log_error)
            LOG_ERROR(
                "Could not find connection: " + name);
    return conn;
}

int Network::get_num_neurons() const {
    int num_neurons = 0;
    for (auto structure : structures)
        num_neurons += structure->get_num_neurons();
    return num_neurons;
}

int Network::get_num_layers() const {
    int num_layers = 0;
    for (auto structure : structures)
        num_layers += structure->get_layers().size();
    return num_layers;
}

int Network::get_num_connections() const {
    int num_connections = 0;
    for (auto structure : structures)
        num_connections += structure->get_connections().size();
    return num_connections;
}

int Network::get_num_weights() const {
    int num_weights = 0;
    for (auto structure : structures)
        for (auto conn : structure->get_connections())
            num_weights += conn->get_num_weights();
    return num_weights;
}

int Network::get_max_layer_size() const {
    int max_size = 0;
    for (auto& structure : this->get_structures()) 
        for (auto& layer : structure->get_layers())
            if (layer->size > max_size) max_size = layer->size;  
    return max_size;
}

void Network::add_structure_internal(StructureConfig *struct_config) {
    for (auto& st : this->structures)
        if (st->name == struct_config->name)
            LOG_ERROR(
                "Repeated structure name: " + st->name);
    this->structures.push_back(new Structure(struct_config));
}

void Network::add_connection_internal(const ConnectionConfig* conn_config) {
    std::string from_structure, to_structure;

    // If there is one structure, omitted structures are fine
    if (structures.size() == 1)
        from_structure = to_structure = structures.at(0)->name;
    else {
        if (not conn_config->has("from structure"))
            LOG_ERROR(
                "Unspecified source structure for connection: "
                + conn_config->get("name", ""));
        if (not conn_config->has("to structure"))
            LOG_ERROR(
                "Unspecified destination structure for connection: "
                + conn_config->get("name", ""));
        from_structure = conn_config->get("from structure");
        to_structure = conn_config->get("to structure");
    }

    Structure::connect(
        get_structure(from_structure),
        get_structure(to_structure),
        conn_config);
}
