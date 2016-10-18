#include <sstream>

#include "model/model.h"

#define cimg_display 0
#include "libs/CImg.h"

Model::Model (std::string driver_string) :
        num_neurons(0),
        driver_string(driver_string) {}

int Model::connect_layers(int from_id, int to_id, bool plastic,
        int delay, float max_weight, ConnectionType type, Opcode opcode,
        std::string params) {
    int conn_id = this->connections.size();
    Connection *conn = new Connection(
        conn_id,
        this->layers[from_id], this->layers[to_id],
        plastic, delay, max_weight, type, params, opcode);
    this->connections.push_back(conn);
    return conn_id;
}

int Model::connect_layers_shared(int from_id, int to_id, int parent_id) {
    // Ensure parent doesn't have a parent
    if (this->connections[parent_id]->parent != NULL)
        throw "Shared connections must refer to non-shared connection!";

    // Ensure that the weights can be shared by checking sizes
    Connection *parent = this->connections[parent_id];
    Layer *from_layer = this->layers[from_id];
    Layer *to_layer = this->layers[to_id];
    int conn_id = this->connections.size();

    if (from_layer->rows == parent->from_layer->rows
            and from_layer->columns == parent->from_layer->columns
            and to_layer->rows == parent->to_layer->rows
            and to_layer->columns == parent->to_layer->columns) {
        Connection *conn = new Connection(
            conn_id, from_layer, to_layer,
            parent);
        this->connections.push_back(conn);
        return conn_id;
    } else {
        throw "Cannot share weights between connections of different sizes!";
    }
}

int Model::connect_layers_expected(int from_id,
        std::string new_layer_params, bool plastic, int delay,
        float max_weight, ConnectionType type, Opcode opcode,
        std::string params) {
    Layer *from_layer = this->layers[from_id];

    // Determine new layer size and create
    int expected_rows = get_expected_dimension(
        from_layer->rows, type, params);
    int expected_columns = get_expected_dimension(
        from_layer->columns, type, params);
    int new_id = add_layer(
        expected_rows, expected_columns, new_layer_params);
    Layer *to_layer = this->layers[new_id];

    // Connect new layer to given layer
    int conn_id = this->connections.size();
    Connection *conn = new Connection(
        conn_id, from_layer, to_layer,
        plastic, delay, max_weight, type, params, opcode);
    this->connections.push_back(conn);

    // Return id of new layer
    return new_id;
}

int Model::add_layer(int rows, int columns, std::string params) {
    // Index of first neuron for layer
    int start_index = this->num_neurons;
    int layer_index = this->layers.size();

    this->layers.push_back(new Layer(layer_index, start_index, rows, columns, params));

    // Add neurons.
    this->add_neurons(rows*columns);

    return layer_index;
}

int Model::add_layer_from_image(std::string path, std::string params) {
    cimg_library::CImg<unsigned char> img(path.c_str());
    return this->add_layer(img.height(), img.width(), params);
}

void Model::add_input(int layer, std::string type, std::string params) {
    InputModule *input_module = build_input(this->layers[layer], type, params);
    this->input_modules.push_back(input_module);
}

void Model::add_output(int layer, std::string type, std::string params) {
    OutputModule *output_module = build_output(this->layers[layer], type, params);
    this->output_modules.push_back(output_module);
}

static bool contains(std::vector<Layer *> layers, Layer* layer) {
    return find(layers.begin(), layers.end(), layer) != layers.end();
}

void Model::rearrange() {
    std::vector<Layer *> input_layers;
    std::vector<Layer *> io_layers;
    std::vector<Layer *> output_layers;
    std::vector<Layer *> new_layers;

    // Push input
    for (int i = 0 ; i < this->input_modules.size(); ++i) {
        Layer *layer = this->input_modules[i]->layer;
        if (!contains(input_layers, layer))
            input_layers.push_back(layer);
    }

    // Push output
    // If contained in input, add to IO list
    for (int i = 0 ; i < this->output_modules.size(); ++i) {
        Layer *layer = this->output_modules[i]->layer;
        if (!contains(output_layers, layer))
            output_layers.push_back(layer);
        if (!contains(io_layers, layer) and contains(input_layers, layer))
            io_layers.push_back(layer);
    }

    // Add input layers if not in IO
    // Add IO layers if not in new
    // Add output layers if not in new
    for (int i = 0 ; i < input_layers.size(); ++i) {
        Layer *layer = input_layers[i];
        if (!contains(io_layers, layer))
            new_layers.push_back(layer);
    }
    for (int i = 0 ; i < io_layers.size(); ++i) {
        Layer *layer = io_layers[i];
        if (!contains(new_layers, layer))
            new_layers.push_back(layer);
    }
    for (int i = 0 ; i < output_layers.size(); ++i) {
        Layer *layer = output_layers[i];
        if (!contains(new_layers, layer))
            new_layers.push_back(layer);
    }

    // Push remaining
    for (int i = 0 ; i < this->layers.size(); ++i) {
        Layer *layer = this->layers[i];
        if (!contains(new_layers, layer))
            new_layers.push_back(layer);
    }

    // Adjust indices and ids
    int start_index = 0;
    for (int i = 0 ; i < new_layers.size(); ++i) {
        new_layers[i]->id = i;
        new_layers[i]->index = start_index;
        start_index += new_layers[i]->size;
    }

    this->layers = new_layers;
}
