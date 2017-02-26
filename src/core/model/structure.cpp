#include "model/structure.h"
#include "util/error_manager.h"

#define cimg_display 0
#include "libs/CImg.h"

Connection* Structure::connect(
        Structure *from_structure, std::string from_layer_name,
        Structure *to_structure, std::string to_layer_name,
        bool plastic, int delay, float max_weight, ConnectionType type,
        Opcode opcode, std::string params) {
    Layer *from_layer = from_structure->find_layer(from_layer_name);
    Layer *to_layer = to_structure->find_layer(to_layer_name);
    if (from_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + from_layer_name + "\"!");
    if (to_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + to_layer_name + "\"!");

    return to_structure->connect_layers(
        from_layer, to_layer,
        plastic, delay, max_weight,
        type, opcode, params);
}

Connection* Structure::connect_layers(
        Layer *from_layer, Layer *to_layer,
        bool plastic, int delay, float max_weight,
        ConnectionType type, Opcode opcode, std::string params) {
    Connection *conn = new Connection(
        from_layer, to_layer,
        plastic, delay, max_weight, type, params, opcode);
    to_layer->add_to_root(conn);
    this->connections.push_back(conn);
    return conn;
}

Connection* Structure::connect_layers(
        std::string from_layer_name, std::string to_layer_name,
        bool plastic, int delay, float max_weight,
        ConnectionType type, Opcode opcode, std::string params) {
    Layer *from_layer = find_layer(from_layer_name);
    Layer *to_layer = find_layer(to_layer_name);
    if (from_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + from_layer_name + "\"!");
    if (to_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + to_layer_name + "\"!");
    return connect_layers(from_layer, to_layer,
        plastic, delay, max_weight,
        type, opcode, params);
}

Connection* Structure::connect_layers_expected(
        std::string from_layer_name, std::string to_layer_name,
        NeuralModel neural_model, std::string new_layer_params,
        bool plastic, int delay,
        float max_weight, ConnectionType type, Opcode opcode,
        std::string params, float noise) {
    Layer *from_layer = find_layer(from_layer_name);
    if (from_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + from_layer_name + "\"!");

    // Determine new layer size and create
    int expected_rows = get_expected_dimension(
        from_layer->rows, type, params);
    int expected_columns = get_expected_dimension(
        from_layer->columns, type, params);
    add_layer(to_layer_name, neural_model,
        expected_rows, expected_columns, new_layer_params, noise);
    Layer *to_layer = find_layer(to_layer_name);

    // Connect new layer to given layer
    Connection *conn = connect_layers(
        from_layer, to_layer,
        plastic, delay, max_weight, type, opcode, params);
    return conn;
}

Connection* Structure::connect_layers_matching(
        std::string from_layer_name, std::string to_layer_name,
        NeuralModel neural_model, std::string new_layer_params,
        bool plastic, int delay,
        float max_weight, ConnectionType type, Opcode opcode,
        std::string params, float noise) {
    Layer *from_layer = find_layer(from_layer_name);
    if (from_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + from_layer_name + "\"!");

    // Determine new layer size and create
    add_layer(to_layer_name, neural_model,
        from_layer->rows, from_layer->columns, new_layer_params, noise);
    Layer *to_layer = find_layer(to_layer_name);

    // Connect new layer to given layer
    Connection *conn = connect_layers(
        from_layer, to_layer,
        plastic, delay, max_weight, type, opcode, params);
    return conn;
}

DendriticNode *Structure::spawn_dendritic_node(std::string to_layer_name) {
    Layer *to_layer = find_layer(to_layer_name);
    if (to_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + to_layer_name + "\"!");
    return to_layer->dendritic_root->add_child();
}

Connection* Structure::connect_layers_internal(
        DendriticNode *node, std::string from_layer_name,
        bool plastic, int delay, float max_weight, ConnectionType type,
        Opcode opcode, std::string params) {
    Layer *from_layer = find_layer(from_layer_name);
    Layer *to_layer = node->to_layer;
    if (from_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + from_layer_name + "\"!");

    Connection *conn = new Connection(
        from_layer, to_layer,
        plastic, delay, max_weight, type, params, opcode);
    node->add_child(conn);
    this->connections.push_back(conn);
    return conn;
}

void Structure::add_layer(std::string name, NeuralModel neural_model,
        int rows, int columns, std::string params, float noise) {
    if (this->layers_by_name.find(name) != this->layers_by_name.end())
        ErrorManager::get_instance()->log_error(
            "Repeated layer name!");

    Layer* layer = new Layer(name, neural_model, this,
        rows, columns, params, noise);
    this->layers.push_back(layer);
    this->layers_by_name[name] = layer;
    this->total_neurons += layer->size;
    this->num_neurons[layer->get_type()] += layer->size;
    this->neural_model_flags[neural_model] = true;
}

void Structure::add_layer_from_image(std::string name, NeuralModel neural_model,
        std::string path, std::string params, float noise) {
    cimg_library::CImg<unsigned char> img(path.c_str());
    this->add_layer(name, neural_model, img.height(), img.width(), params, noise);
}

void Structure::add_module(std::string layer_name,
        std::string type, std::string params) {
    Layer *layer = find_layer(layer_name);
    if (layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + layer_name + "\"!");

    Module *module = build_module(layer, type, params);

    // Remove neurons from old IOType
    this->num_neurons[layer->get_type()] -= layer->size;

    // Add the module
    layer->add_module(module);

    // Add neurons to new IOType
    this->num_neurons[layer->get_type()] += layer->size;
}
