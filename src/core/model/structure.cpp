#include "model/structure.h"
#include "util/error_manager.h"

#define cimg_display 0
#include "libs/CImg.h"

Structure::Structure (std::string name) : name(name) { }

Connection* Structure::connect(
        Structure *from_structure, std::string from_layer_name,
        Structure *to_structure, std::string to_layer_name,
        bool plastic, int delay, float max_weight, ConnectionType type,
        Opcode opcode, std::string params) {
    Layer *from_layer = from_structure->find_layer(from_layer_name);
    Layer *to_layer = to_structure->find_layer(from_layer_name);
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
    int conn_id = this->connections.size();
    Connection *conn = new Connection(
        conn_id, from_layer, to_layer,
        plastic, delay, max_weight, type, params, opcode);
    to_layer->add_to_root(conn);
    this->connections.push_back(conn);
    return conn;
}

Connection* Structure::connect_layers(
        Layer *from_layer, Layer *to_layer,
        Connection *parent) {
    int conn_id = this->connections.size();
    Connection *conn = new Connection(
        conn_id, from_layer, to_layer, parent);
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

Connection* Structure::connect_layers_shared(
        std::string from_layer_name, std::string to_layer_name,
        Connection* parent) {
    // If parent has a parent, follow up and find the original connection
    while (parent->parent != NULL)
        parent = parent->parent;

    Layer *from_layer = find_layer(from_layer_name);
    Layer *to_layer = find_layer(to_layer_name);
    if (from_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + from_layer_name + "\"!");
    if (to_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + to_layer_name + "\"!");

    // Ensure that the weights can be shared by checking sizes
    if (from_layer->rows == parent->from_layer->rows
            and from_layer->columns == parent->from_layer->columns
            and to_layer->rows == parent->to_layer->rows
            and to_layer->columns == parent->to_layer->columns) {
        return connect_layers(
            from_layer, to_layer, parent);
    } else {
        ErrorManager::get_instance()->log_error(
            "Cannot share weights between connections of different sizes!");
    }
}

Connection* Structure::connect_layers_expected(
        std::string from_layer_name, std::string to_layer_name,
        std::string new_layer_params, bool plastic, int delay,
        float max_weight, ConnectionType type, Opcode opcode,
        std::string params) {
    Layer *from_layer = find_layer(from_layer_name);
    if (from_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + from_layer_name + "\"!");

    // Determine new layer size and create
    int expected_rows = get_expected_dimension(
        from_layer->rows, type, params);
    int expected_columns = get_expected_dimension(
        from_layer->columns, type, params);
    add_layer(to_layer_name, expected_rows, expected_columns, new_layer_params);
    Layer *to_layer = find_layer(to_layer_name);

    // Connect new layer to given layer
    Connection *conn = connect_layers(
        from_layer, to_layer,
        plastic, delay, max_weight, type, opcode, params);
    return conn;
}

Connection* Structure::connect_layers_matching(
        std::string from_layer_name, std::string to_layer_name,
        std::string new_layer_params, bool plastic, int delay,
        float max_weight, ConnectionType type, Opcode opcode,
        std::string params) {
    Layer *from_layer = find_layer(from_layer_name);
    if (from_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + from_layer_name + "\"!");

    // Determine new layer size and create
    add_layer(to_layer_name, from_layer->rows, from_layer->columns, new_layer_params);
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

Connection* Structure::connect_layers_internal(DendriticNode *node,
        std::string from_layer_name, std::string to_layer_name,
        bool plastic, int delay, float max_weight, ConnectionType type,
        Opcode opcode, std::string params) {
    Layer *from_layer = find_layer(from_layer_name);
    Layer *to_layer = find_layer(to_layer_name);
    if (from_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + from_layer_name + "\"!");
    if (to_layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + to_layer_name + "\"!");

    int conn_id = this->connections.size();
    Connection *conn = new Connection(
        conn_id, from_layer, to_layer,
        plastic, delay, max_weight, type, params, opcode);
    node->add_child(conn);
    this->connections.push_back(conn);
    return conn;
}

void Structure::add_layer(std::string name, int rows, int columns, std::string params) {
    if (this->layers_by_name.find(name) != this->layers_by_name.end())
        ErrorManager::get_instance()->log_error(
            "Repeated layer name!");

    Layer* layer = new Layer(name, rows, columns, params);
    this->layers.push_back(layer);
    this->layers_by_name[name] = layer;
}

void Structure::add_layer_from_image(std::string name, std::string path, std::string params) {
    cimg_library::CImg<unsigned char> img(path.c_str());
    this->add_layer(name, img.height(), img.width(), params);
}

void Structure::add_module(std::string layer_name, std::string type, std::string params) {
    Layer *layer = find_layer(layer_name);
    if (layer == NULL)
        ErrorManager::get_instance()->log_error(
            "Could not find layer \"" + layer_name + "\"!");

    Module *module = build_module(layer, type, params);
    layer->add_module(module);
}
