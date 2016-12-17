#include "model/model.h"
#include "model/model_builder.h"
#include "util/error_manager.h"

Model::Model (std::string engine_name) :
        num_neurons(0),
        engine_name(engine_name) {}

Model* Model::load(std::string path) {
    return load_model(path);
}

void Model::add_structure(Structure *structure) {
    if (this->structures.find(structure->name) != this->structures.end())
        ErrorManager::get_instance()->log_error(
            "Repeated structure name!");
    this->structures[structure->name] = structure;
    build();
}

void Model::build() {
    all_layers.clear();
    connections.clear();
    this->num_neurons = 0;
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        layers[i].clear();

    // Extract layers and connections from structures
    int conn_id = 0;
    for (auto& it : this->structures) {
        Structure *structure = it.second;
        all_layers.insert(all_layers.end(),
            structure->layers.begin(), structure->layers.end());

        for (auto& conn : structure->connections) {
            connections.push_back(conn);
            conn->id = conn_id++;
        }
    }

    // Sort layers
    for (auto& layer : this->all_layers) {
        layers[layer->type].push_back(layer);
        this->num_neurons += layer->size;
    }

    // Clear old list
    // Add in order: input, IO, output, internal
    all_layers.clear();
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        all_layers.insert(this->all_layers.end(),
            layers[i].begin(), layers[i].end());

    // Adjust indices and ids
    int start_index = 0;
    int index = 0;
    for (auto& layer : this->all_layers) {
        layer->id = index;
        layer->start_index = start_index;
        start_index += layer->size;
        ++index;
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
