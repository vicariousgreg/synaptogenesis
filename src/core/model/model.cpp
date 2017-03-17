#include "model/model.h"
#include "model/model_builder.h"
#include "util/error_manager.h"

Model::~Model() {
    for (auto& structure : this->structures) delete structure;
}

Model* Model::load(std::string path) {
    return load_model(path);
}

Structure* Model::add_structure(std::string name, ClusterType cluster_type) {
    for (auto& st : this->structures)
        if (st->name == name)
            ErrorManager::get_instance()->log_error(
                "Repeated structure name!");
    Structure *structure = new Structure(name, cluster_type);
    this->structures.push_back(structure);
    return structure;
}

LayerList Model::get_layers() const {
    LayerList layers;
    for (auto& structure : structures)
        for (auto& layer : structure->get_layers())
            layers.push_back(layer);
    return layers;
}

LayerList Model::get_layers(NeuralModel neural_model) const {
    LayerList layers;
    for (auto& structure : structures)
        for (auto& layer : structure->get_layers())
            if (layer->neural_model == neural_model)
                layers.push_back(layer);
    return layers;
}

LayerList Model::get_input_layers() const {
    LayerList layers;
    for (auto layer : this->get_layers())
        if (layer->is_input()) layers.push_back(layer);
    return layers;
}

LayerList Model::get_output_layers() const {
    LayerList layers;
    for (auto layer : this->get_layers())
        if (layer->is_output()) layers.push_back(layer);
    return layers;
}

LayerList Model::get_expected_layers() const {
    LayerList layers;
    for (auto layer : this->get_layers())
        if (layer->is_expected()) layers.push_back(layer);
    return layers;
}

int Model::get_num_neurons() const {
    int num_neurons = 0;
    for (auto structure : structures)
        num_neurons += structure->get_num_neurons();
    return num_neurons;
}

int Model::get_num_layers() const {
    int num_layers = 0;
    for (auto structure : structures)
        num_layers += structure->get_layers().size();
    return num_layers;
}

int Model::get_num_connections() const {
    int num_connections = 0;
    for (auto structure : structures)
        num_connections += structure->get_connections().size();
    return num_connections;
}

int Model::get_num_weights() const {
    int num_weights = 0;
    for (auto structure : structures)
        for (auto conn : structure->get_connections())
            num_weights += conn->get_num_weights();
    return num_weights;
}
