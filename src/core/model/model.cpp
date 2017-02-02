#include "model/model.h"
#include "model/model_builder.h"
#include "util/error_manager.h"

Model::Model(std::string engine_name) :
        num_neurons(0),
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
    this->num_neurons = 0;
    for (auto type : IOTypes)
        layers[type].clear();

    // Extract layers and connections from structures
    for (auto& it : this->structures) {
        Structure *structure = it.second;
        all_layers.insert(all_layers.end(),
            structure->layers.begin(), structure->layers.end());
        connections.insert(connections.end(),
            structure->connections.begin(), structure->connections.end());
    }

    // Sort layers
    for (auto& layer : this->all_layers) {
        layers[layer->type].push_back(layer);
        this->num_neurons += layer->size;
    }

    // Clear old list
    // Add in order: input, IO, output, internal
    all_layers.clear();
    for (auto type : IOTypes)
        all_layers.insert(this->all_layers.end(),
            layers[type].begin(), layers[type].end());

    // Adjust indices and ids
    int start_index = 0;
    for (auto& layer : this->all_layers) {
        layer->start_index = start_index;
        start_index += layer->size;
    }

    // Set input and output indices
    int input_index = 0;
    for (auto& layer : this->layers[INPUT]) {
        layer->input_index = input_index;
        input_index += layer->size;
    }
    int output_index = 0;
    for (auto& layer : this->layers[INPUT_OUTPUT]) {
        layer->input_index = input_index;
        layer->output_index = output_index;
        input_index += layer->size;
        output_index += layer->size;
    }
    for (auto& layer : this->layers[OUTPUT]) {
        layer->output_index = output_index;
        output_index += layer->size;
    }
}
