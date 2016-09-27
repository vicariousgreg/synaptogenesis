#ifndef model_h
#define model_h

#include <vector>
#include <string>

#include "constants.h"

class Layer {
    public:
        /* Constructor.
         * |start_index| identifies the first neuron in the layer.
         * |size| indicates the size of the layer.
         */
        Layer(int layer_id, int start_index, int size) :
                id(layer_id),
                index(start_index),
                size(size) {}

        // Layer ID
        int id;

        // Index of first neuron
        int index;

        // Size of layer
        int size;
};

class Connection {
    public:
        Connection (int conn_id, Layer &from_layer, Layer &to_layer, bool plastic,
                int delay, float max_weight, ConnectionType type, OPCODE opcode) :
                    id(conn_id),
                    from_layer(from_layer),
                    to_layer(to_layer),
                    plastic(plastic),
                    delay(delay),
                    max_weight(max_weight),
                    opcode(opcode),
                    type(type),
                    parent(-1) {
            if (delay > (32 * HISTORY_SIZE - 1))
                throw "Cannot implement connection delay longer than history!";

            if (type == FULLY_CONNECTED) {
                this->num_weights = from_layer.size * to_layer.size;
            } else if (type == ONE_TO_ONE) {
                if (from_layer.size != to_layer.size) {
                    throw "Cannot connect differently sized layers one-to-one!";
                } else {
                    this->num_weights = from_layer.size;
                }
            }
        }

        Connection(int conn_id, Layer &from_layer, Layer &to_layer, int parent) :
                id(conn_id),
                from_layer(from_layer),
                to_layer(to_layer),
                parent(parent) {
            if (type == FULLY_CONNECTED) {
                this->num_weights = from_layer.size * to_layer.size;
            } else if (type == ONE_TO_ONE) {
                this->num_weights = from_layer.size;
            }
        }

        // Connection ID
        int id;

        // Matrix type (see enum)
        ConnectionType type;

        // Associated layers
        Layer from_layer, to_layer;

        // Number of weights in connection
        int num_weights;

        // Connection operation code
        OPCODE opcode;

        // Connection delay
        int delay;

        // Flag for whether matrix can change via learning
        bool plastic;

        // Maximum weight for randomization
        float max_weight;

        // ID of parent matrix if this is a shared connection
        int parent;
};

class Model {
    public:
        Model ();

        /* Adds a layer to the environment with the given parameters */
        int add_layer(int size, std::string params);

        /* Connects two layers, creating a weight matrix with the given 
         *   parameters */
        int connect_layers(int from_layer, int to_layer, bool plastic,
            int delay, float max_weight, ConnectionType type, OPCODE opcode);

        /* Connects to layers, sharing weights with another connection
         *   specified by |parent_id| */
        int connect_layers_shared(int from_layer, int to_layer, int parent_id);

        /* Adds a single neuron.
         * This is called from add_layer() */
        int add_neuron(std::string params);

        // Neurons
        int num_neurons;

        // Layers
        int num_layers;
        std::vector<Layer> layers;

        // Connection matrices.
        int num_connections;
        std::vector<Connection> connections;

        // Parameter Vector
        std::vector<std::string> neuron_parameters;
};

#endif
