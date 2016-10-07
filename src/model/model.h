#ifndef model_h
#define model_h

#include <vector>
#include <string>

#include "constants.h"

class Input;
class Output;

class Layer {
    public:
        /* Constructor.
         * |start_index| identifies the first neuron in the layer.
         * |size| indicates the size of the layer.
         */
        Layer(int layer_id, int start_index, int rows, int columns) :
                id(layer_id),
                index(start_index),
                rows(rows),
                columns(columns),
                size(rows * columns),
                input(NULL),
                output(NULL) {}

        // Layer ID and start index
        int id, index;

        // Layer rows, columns, and total size
        int rows, columns, size;

        // Parameters for initializing connection
        // Some types will parse values for connection construction
        //   -> Divergent, Convergent, Convolutional
        // In this case, the constructor will consume these values and leave
        //   the remaining values here
        std::string params;

        // Input driver
        // If none, this will be null
        Input* input;

        // Output driver
        // If none, this will be null
        Output* output;
};

/* Gets the expected row/col size of a destination layer given a |source_layer|,
 *   a connection |type| and connection |params|.
 * This function only returns meaningful values for connection types that
 *   are not FULLY_CONNECTED */
int get_expected_dimension(int dimension, ConnectionType type, std::string params);

class Connection {
    public:
        Connection (int conn_id, Layer *from_layer, Layer *to_layer, bool plastic,
                int delay, float max_weight, ConnectionType type,
                std::string params, Opcode opcode);

        Connection(int conn_id, Layer *from_layer, Layer *to_layer,
                Connection *parent);

        // Connection ID
        // ID of parent matrix if this is a shared connection
        int id, parent;

        // Layer parameters
        std::string params;

        // Matrix type (see enum)
        ConnectionType type;

        // Convolutional boolean (extracted from type)
        bool convolutional;

        // Overlap and stride, if relevant
        int overlap, stride;

        // Extracted values.
        int from_index, to_index;
        int from_size, from_rows, from_columns;
        int to_size, to_rows, to_columns;

        // Number of weights in connection
        // Connection delay
        int num_weights, delay;

        // Connection operation code
        Opcode opcode;

        // Flag for whether matrix can change via learning
        bool plastic;

        // Maximum weight
        float max_weight;
};

class Model {
    public:
        Model (std::string driver_string);

        /* Adds a layer to the environment with the given parameters */
        int add_layer(int rows, int columns, std::string params);

        /* Connects two layers, creating a weight matrix with the given 
         *   parameters */
        int connect_layers(int from_id, int to_id, bool plastic,
            int delay, float max_weight, ConnectionType type, Opcode opcode,
            std::string params);

        /* Connects to layers, sharing weights with another connection
         *   specified by |parent_id| */
        int connect_layers_shared(int from_id, int to_id, int parent_id);

        /* Adds an input hook of the given |type| for the given |layer| */
        void add_input(int layer, std::string type, std::string params);

        /* Adds an output hook of the given |type| for the given |layer| */
        void add_output(int layer, std::string type, std::string params);

        // Driver string indicating type of driver
        std::string driver_string;

        // Neurons
        int num_neurons;

        // Layers
        int num_layers;
        std::vector<Layer*> layers;

        // Connection matrices
        int num_connections;
        std::vector<Connection*> connections;

        // Parameter strings vector
        std::vector<std::string> parameter_strings;

    private:
        /* Adds a single neuron.
         * This is called from add_layer() */
        void add_neurons(int count, std::string params) {
            for (int i = 0; i < count; ++i)
                this->parameter_strings.push_back(params);
            this->num_neurons += count;
        }
};

#endif
