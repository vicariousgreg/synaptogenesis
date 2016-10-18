#ifndef model_h
#define model_h

#include <vector>
#include <string>

#include "constants.h"
#include "model/layer.h"
#include "model/connection.h"

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

        /* Reorganizes the model to place input and ouput layers in
         *   contiguous memory. */
        void rearrange();

        // Driver string indicating type of driver
        std::string driver_string;

        // Total number of neurons
        int num_neurons;

        // Layers
        std::vector<Layer*> layers;

        // Connections
        std::vector<Connection*> connections;

        // Input and output modules
        std::vector<Input*> input_modules;
        std::vector<Output*> output_modules;

    private:
        /* Adds a given number of neurons.
         * This is called from add_layer() */
        void add_neurons(int count) {
            this->num_neurons += count;
        }
};

#endif
