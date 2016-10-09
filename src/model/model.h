#ifndef model_h
#define model_h

#include <vector>
#include <string>

#include "constants.h"

/* Forward declarations for input and output modules */
class Input;
class Output;

/* Represents a two dimensional layer of neurons.
 * Layers can be constructed and connected into networks using the Model class.
 *
 * Layers contain:
 *   - unique identifier
 *   - starting index in the neural arrays
 *   - size information
 *   - parameters for matrix initialization
 *
 */
class Layer {
    public:
        // Layer ID and start index
        int id, index;

        // Layer rows, columns, and total size
        int rows, columns, size;

        // Parameters for initializing neural properties
        std::string params;

    private:
        friend class Model;

        /* Constructor.
         * |start_index| identifies the first neuron in the layer.
         * |size| indicates the size of the layer.
         */
        Layer(int layer_id, int start_index, int rows, int columns) :
                id(layer_id),
                index(start_index),
                rows(rows),
                columns(columns),
                size(rows * columns) {}
};

/* Gets the expected row/col size of a destination layer given a |source_layer|,
 *   a connection |type| and connection |params|.
 * This function only returns meaningful values for connection types that
 *   are not FULLY_CONNECTED, because they can link layers of any sizes */
int get_expected_dimension(int source_val, ConnectionType type,
                                            std::string params);

/* Represents a connection between two neural layrs.
 * Connections bridge Layers and are constructed in the Model class.
 * Connections have several types, enumerated and documented in "constants.h".
 *
 * Connections contain:
 *   - connection type enum
 *   - unique identifier
 *   - parent id if sharing weights with another connection
 *   - convolutional flag if sharing weights internally
 *   - arborization parameters (if applicable)
 *   - parameters for matrix initialization
 *   - total number of actual weights in the connection
 *   - extracted layer properties
 *   - connection delay
 *   - connection opcode (see constants.h)
 *   - plasticity boolean
 *   - maximum weight value
 *
 */
class Connection {
    public:
        // Matrix type
        ConnectionType type;

        // Connection ID
        // ID of parent matrix if this is a shared connection
        int id, parent;

        // Convolutional boolean (extracted from type)
        bool convolutional;

        // Arborization parameters (extracted from params)
        // The amount of overlap and stride for arborized
        //   (convergent/divergent) connections
        int overlap, stride;

        // Parameters for matrix construction
        // Some types will parse values for connection construction
        //   -> Divergent, Convergent, Convolutional
        // In this case, the constructor will consume these values and leave
        //   the remaining values here
        std::string params;

        // Extracted Layer properties
        int from_index, to_index;
        int from_size, from_rows, from_columns;
        int to_size, to_rows, to_columns;

        // Number of weights in connection
        int num_weights;
        // Connection delay
        int delay;

        // Connection operation code
        Opcode opcode;

        // Flag for whether matrix can change via learning
        bool plastic;

        // Maximum for weights
        float max_weight;

    private:
        friend class Model;

        Connection (int conn_id, Layer *from_layer, Layer *to_layer,
                bool plastic, int delay, float max_weight,
                ConnectionType type, std::string params, Opcode opcode);

        Connection(int conn_id, Layer *from_layer, Layer *to_layer,
                Connection *parent);

};

/* Represents a full neural network model.
 * Contains network graph data and parameters for layers/connections.
 *
 * Layers can be created using add_layer()
 * Layers can be connected in a few ways:
 *   - connect_layers() connects two layers with given parameters
 *   - connect_layers_shared() connects two layers, sharing weights with
 *       another connection and inheriting its properties
 *   - connect_layers_expected() extrapolates destination layer size given
 *       a source layer and connection parameters
 *
 *  In addition, input and output modules can be attached to layers using
 *    add_input() and add_output().  These modules contain hooks for driving
 *    input to a layer or extracting and using output of a layer.
 *
 */
class Model {
    public:
        Model (std::string driver_string);

        /* Adds a layer to the environment with the given parameters */
        int add_layer(int rows, int columns, std::string params);
        int add_layer_from_image(std::string path, std::string params);

        /* Connects two layers, creating a weight matrix with the given 
         *   parameters */
        int connect_layers(int from_id, int to_id, bool plastic,
            int delay, float max_weight, ConnectionType type, Opcode opcode,
            std::string params);

        /* Connects to layers, sharing weights with another connection
         *   specified by |parent_id| */
        int connect_layers_shared(int from_id, int to_id, int parent_id);

        /* Uses expected sizes to create a new layer and connect it to the
         *   given layer.  Returns the id of the new layer. */
        int connect_layers_expected(int from_id, std::string new_layer_params,
                bool plastic, int delay, float max_weight,
                ConnectionType type, Opcode opcode, std::string params);

        /* Adds an input module of the given |type| for the given |layer| */
        void add_input(int layer, std::string type, std::string params);

        /* Adds an output module of the given |type| for the given |layer| */
        void add_output(int layer, std::string type, std::string params);

        // Driver string indicating type of driver
        std::string driver_string;

        // Total number of neurons
        int num_neurons;

        // Layers
        std::vector<Layer*> layers;

        // Connections
        std::vector<Connection*> connections;

        // Parameter strings vector
        std::vector<std::string> parameter_strings;

        // Input and output modules
        std::vector<Input*> input_modules;
        std::vector<Output*> output_modules;

    private:
        /* Adds a given number of neurons.
         * This is called from add_layer() */
        void add_neurons(int count, std::string params) {
            for (int i = 0; i < count; ++i)
                this->parameter_strings.push_back(params);
            this->num_neurons += count;
        }
};

#endif
