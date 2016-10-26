#include "model/model.h"
#include "error_manager.h"

Model::Model (std::string engine_name) :
        num_neurons(0),
        engine_name(engine_name) {}

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
    for (auto it = structures.begin(); it != structures.end(); ++it) {
        Structure *structure = it->second;
        all_layers.insert(all_layers.end(),
            structure->layers.begin(), structure->layers.end());

        for (auto it = structure->connections.begin(); it != structure->connections.end(); ++it) {
            connections.push_back(*it);
            (*it)->id = conn_id++;
        }
    }

    // Sort layers
    for (int i = 0 ; i < this->all_layers.size(); ++i) {
        Layer *layer = this->all_layers[i];
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
    for (int i = 0 ; i < all_layers.size(); ++i) {
        all_layers[i]->index = start_index;
        start_index += all_layers[i]->size;
    }

    // Set input and output indices
    int input_index = 0;
    for (int i = 0 ; i < layers[INPUT].size(); ++i) {
        layers[INPUT][i]->input_index = input_index;
        input_index += layers[INPUT][i]->size;
    }
    int output_index = 0;
    for (int i = 0 ; i < layers[INPUT_OUTPUT].size(); ++i) {
        layers[INPUT_OUTPUT][i]->input_index = input_index;
        layers[INPUT_OUTPUT][i]->output_index = output_index;
        input_index += layers[INPUT_OUTPUT][i]->size;
        output_index += layers[INPUT_OUTPUT][i]->size;
    }
    for (int i = 0 ; i < layers[OUTPUT].size(); ++i) {
        layers[OUTPUT][i]->output_index = output_index;
        output_index += layers[OUTPUT][i]->size;
    }
}
