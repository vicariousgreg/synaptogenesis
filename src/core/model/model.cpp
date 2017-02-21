#include "model/model.h"
#include "model/model_builder.h"
#include "util/error_manager.h"

Model::Model(std::string engine_name) :
        total_neurons(0),
        engine_name(engine_name) {}

Model::~Model() {
    for (auto& conn : this->connections) delete conn;
    for (auto& layer : this->all_layers) delete layer;
    for (auto& structure : this->structures) delete structure.second;
}

Model* Model::load(std::string path) {
    return load_model(path);
}

void Model::add_structure(Structure *structure) {
    if (this->structures.find(structure->name) != this->structures.end())
        ErrorManager::get_instance()->log_error(
            "Repeated structure name!");
    this->structures[structure->name] = structure;
    this->build();
}

void Model::build() {
    all_layers.clear();
    connections.clear();
    this->total_neurons = 0;
    for (auto type : IOTypes) this->num_neurons[type] = 0;
    for (auto type : IOTypes) layers[type].clear();

    // Extract layers and connections from structures
    for (auto& it : this->structures) {
        Structure *structure = it.second;
        auto layers = structure->get_layers();
        auto conns = structure->get_connections();

        all_layers.insert(all_layers.end(), layers.begin(), layers.end());
        connections.insert(this->connections.end(), conns.begin(), conns.end());
    }

    // Sort layers
    for (auto& layer : this->all_layers) {
        layers[layer->type].push_back(layer);
        this->total_neurons += layer->size;
        this->num_neurons[layer->type] += layer->size;
    }

    // Clear old list
    // Add in order: input, IO, output, internal
    all_layers.clear();
    for (auto type : IOTypes)
        all_layers.insert(this->all_layers.end(),
            layers[type].begin(), layers[type].end());
}
