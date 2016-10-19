#include "model/model.h"

#define cimg_display 0
#include "libs/CImg.h"

Model::Model (std::string driver_string) :
        num_neurons(0),
        driver_string(driver_string) {}

Connection* Model::connect_layers(Layer* from_layer, Layer* to_layer,
        bool plastic, int delay, float max_weight,
        ConnectionType type, Opcode opcode, std::string params) {
    int conn_id = this->connections.size();
    Connection *conn = new Connection(
        conn_id, from_layer, to_layer,
        plastic, delay, max_weight, type, params, opcode);
    this->connections.push_back(conn);
    from_layer->add_output_connection(conn);
    to_layer->add_input_connection(conn);
    return conn;
}

Connection* Model::connect_layers_shared(
        Layer* from_layer, Layer* to_layer, Connection* parent) {
    // Ensure parent doesn't have a parent
    if (parent->parent != NULL)
        throw "Shared connections must refer to non-shared connection!";

    // Ensure that the weights can be shared by checking sizes
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
        return conn;
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

    // Add neurons.
    this->add_neurons(rows*columns);

    return layer;
}

Layer* Model::add_layer_from_image(std::string path, std::string params) {
    cimg_library::CImg<unsigned char> img(path.c_str());
    return this->add_layer(img.height(), img.width(), params);
}

void Model::add_module(Layer* layer, std::string type, std::string params) {
    Module *module = build_module(layer, type, params, this->driver_string);
    layer->add_module(module->get_type());
    this->modules.push_back(module);
}

static bool contains(std::vector<Layer *> layers, Layer* layer) {
    return find(layers.begin(), layers.end(), layer) != layers.end();
}

void Model::sort_layers() {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        layers[i].clear();

    // Sort layers
    for (int i = 0 ; i < this->all_layers.size(); ++i) {
        Layer *layer = this->all_layers[i];
        layers[layer->type].push_back(layer);
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
        all_layers[i]->id = i;
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
