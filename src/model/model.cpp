#include <sstream>

#include "model/model.h"
#include "io/input.h"
#include "io/output.h"

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
        std::string new_layer_params,bool plastic, int delay,
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

    this->layers.push_back(new Layer(layer_index, start_index, rows, columns));

    // Add neurons.
    this->add_neurons(rows*columns, params);

    return layer_index;
}

int Model::add_layer_from_image(std::string path, std::string params) {
    cimg_library::CImg<unsigned char> img(path.c_str());
    return this->add_layer(img.height(), img.width(), params);
}

void Model::add_input(int layer, std::string type, std::string params) {
    Input *input = build_input(this->layers[layer], type, params);
    this->input_modules.push_back(input);
}

void Model::add_output(int layer, std::string type, std::string params) {
    Output *output = build_output(this->layers[layer], type, params);
    this->output_modules.push_back(output);
}

void Model::rearrange() {
    std::vector<Layer *> new_layers;

    // Push input
    for (int i = 0 ; i < this->input_modules.size(); ++i)
        new_layers.push_back(this->input_modules[i]->layer);
    // Push output
    for (int i = 0 ; i < this->output_modules.size(); ++i)
        new_layers.push_back(this->output_modules[i]->layer);
    // Push remaining
    for (int i = 0 ; i < this->layers.size(); ++i)
        if (find(new_layers.begin(), new_layers.end(), this->layers[i])
            != new_layers.end())
            new_layers.push_back(this->layers[i]);
}
