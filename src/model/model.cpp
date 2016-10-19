#include <sstream>

#include "model/model.h"

#define cimg_display 0
#include "libs/CImg.h"

Model::Model (std::string driver_string) :
        num_neurons(0),
        driver_string(driver_string) {}

int Model::connect_layers(Layer* from_layer, Layer* to_layer, bool plastic,
        int delay, float max_weight, ConnectionType type, Opcode opcode,
        std::string params) {
    int conn_id = this->connections.size();
    Connection *conn = new Connection(
        conn_id, from_layer, to_layer,
        plastic, delay, max_weight, type, params, opcode);
    this->connections.push_back(conn);
    from_layer->add_output_connection(conn);
    to_layer->add_input_connection(conn);
    return conn_id;
}

int Model::connect_layers_shared(Layer* from_layer, Layer* to_layer, int parent_id) {
    // Ensure parent doesn't have a parent
    if (this->connections[parent_id]->parent != NULL)
        throw "Shared connections must refer to non-shared connection!";

    // Ensure that the weights can be shared by checking sizes
    Connection *parent = this->connections[parent_id];
    int conn_id = this->connections.size();

    if (from_layer->rows == parent->from_layer->rows
            and from_layer->columns == parent->from_layer->columns
            and to_layer->rows == parent->to_layer->rows
            and to_layer->columns == parent->to_layer->columns) {
        Connection *conn = new Connection(
            conn_id, from_layer, to_layer,
            parent);
        this->connections.push_back(conn);
        from_layer->add_output_connection(conn);
        to_layer->add_input_connection(conn);
        return conn_id;
    } else {
        throw "Cannot share weights between connections of different sizes!";
    }
}

Layer* Model::connect_layers_expected(Layer* from_layer,
        std::string new_layer_params, bool plastic, int delay,
        float max_weight, ConnectionType type, Opcode opcode,
        std::string params) {
    // Determine new layer size and create
    int expected_rows = get_expected_dimension(
        from_layer->rows, type, params);
    int expected_columns = get_expected_dimension(
        from_layer->columns, type, params);
    Layer *to_layer = add_layer(
        expected_rows, expected_columns, new_layer_params);

    // Connect new layer to given layer
    int conn_id = this->connections.size();
    Connection *conn = new Connection(
        conn_id, from_layer, to_layer,
        plastic, delay, max_weight, type, params, opcode);
    this->connections.push_back(conn);
    from_layer->add_output_connection(conn);
    to_layer->add_input_connection(conn);

    // Return new layer
    return to_layer;
}

Layer* Model::add_layer(int rows, int columns, std::string params) {
    // Index of first neuron for layer
    int start_index = this->num_neurons;
    int layer_index = this->all_layers.size();

    Layer* layer = new Layer(layer_index, start_index, rows, columns, params);
    this->all_layers.push_back(layer);
    this->layers[INTERNAL].push_back(layer);

    // Add neurons.
    this->add_neurons(rows*columns);

    return layer;
}

Layer* Model::add_layer_from_image(std::string path, std::string params) {
    cimg_library::CImg<unsigned char> img(path.c_str());
    return this->add_layer(img.height(), img.width(), params);
}

void Model::add_input_module(Layer* layer, std::string type, std::string params) {
    if (layer->has_input_module) throw "Layer already has input module!";
    layer->has_input_module = true;

    InputModule *input_module = build_input(layer, type, params);
    this->input_modules.push_back(input_module);
    this->sort_layers();
}

void Model::add_output_module(Layer* layer, std::string type, std::string params) {
    if (layer->has_output_module) throw "Layer already has input module!";
    layer->has_output_module = true;

    OutputModule *output_module = build_output(layer, type, params);
    this->output_modules.push_back(output_module);
    this->sort_layers();
}

static bool contains(std::vector<Layer *> layers, Layer* layer) {
    return find(layers.begin(), layers.end(), layer) != layers.end();
}

void Model::sort_layers() {
    layers[INPUT].clear();
    layers[INPUT_OUTPUT].clear();
    layers[OUTPUT].clear();
    layers[INTERNAL].clear();

    // Sort layers
    for (int i = 0 ; i < this->all_layers.size(); ++i) {
        Layer *layer = this->all_layers[i];
        if (layer->has_input_module and layer->has_output_module)
            layers[INPUT_OUTPUT].push_back(layer);
        else if (layer->has_input_module)
            layers[INPUT].push_back(layer);
        else if (layer->has_output_module)
            layers[OUTPUT].push_back(layer);
        else
            layers[INTERNAL].push_back(layer);
    }

    // Clear old list
    // Add in order: input, IO, output, internal
    all_layers.clear();
    all_layers.insert(this->all_layers.end(), layers[INPUT].begin(), layers[INPUT].end());
    all_layers.insert(this->all_layers.end(), layers[INPUT_OUTPUT].begin(), layers[INPUT_OUTPUT].end());
    all_layers.insert(this->all_layers.end(), layers[OUTPUT].begin(), layers[OUTPUT].end());
    all_layers.insert(this->all_layers.end(), layers[INTERNAL].begin(), layers[INTERNAL].end());

    // Adjust indices and ids
    int start_index = 0;
    for (int i = 0 ; i < all_layers.size(); ++i) {
        all_layers[i]->id = i;
        all_layers[i]->index = start_index;
        start_index += all_layers[i]->size;
    }
}
