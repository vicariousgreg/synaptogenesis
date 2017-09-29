#include "network/network.h"
#include "util/error_manager.h"
#include "builder.h"

Network::~Network() {
    for (auto& structure : structures) delete structure;
}

Network* Network::load(std::string path) {
    return load_network(path);
}

void Network::save(std::string path) {
    save_network(this, path);
}

void Network::add_structure(Structure *structure) {
    for (auto& st : this->structures)
        if (st->name == structure->name)
            ErrorManager::get_instance()->log_error(
                "Repeated structure name: " + st->name);
    this->structures.push_back(structure);
}

Structure* Network::get_structure(std::string name, bool log_error) {
    Structure *structure = nullptr;
    for (auto s : this->structures)
        if (s->name == name)
            structure = s;
    if (structure == nullptr and log_error)
            ErrorManager::get_instance()->log_error(
                "Could not find structure: " + name);
    return structure;
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
