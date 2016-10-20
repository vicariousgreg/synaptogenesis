#ifndef model_h
#define model_h

#include <vector>
#include <string>

#include "constants.h"
#include "model/layer.h"
#include "model/connection.h"
#include "io/module.h"

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
 *    add_module(). These modules contain hooks for driving input to a layer
 *    and extracting and using output of a layer.
 */
class Model {
    public:
        Model (std::string driver_string);
        virtual ~Model() {
            for (int i = 0; i < this->connections.size(); ++i)
                delete this->connections[i];
            for (int i = 0; i < this->all_layers.size(); ++i)
                delete this->all_layers[i];
            for (int i = 0; i < this->modules.size(); ++i)
                delete this->modules[i];
        }

        /* Adds a layer to the environment with the given parameters */
        Layer* add_layer(int rows, int columns, std::string params);
        Layer* add_layer_from_image(std::string path, std::string params);

        /* Connects two layers, creating a weight matrix with the given
         *   parameters */
        Connection* connect_layers(Layer* from_layer, Layer* to_layer,
            bool plastic, int delay, float max_weight, ConnectionType type,
            Opcode opcode, std::string params);

        /* Connects to layers, sharing weights with another connection
         *   specified by |parent_id| */
        Connection* connect_layers_shared(
            Layer* from_layer, Layer* to_layer, Connection* parent);

        /* Uses expected sizes to create a new layer and connect it to the
         *   given layer.  Returns the id of the new layer. */
        Layer* connect_layers_expected(Layer *from_layer,
            std::string new_layer_params,
            bool plastic, int delay, float max_weight,
            ConnectionType type, Opcode opcode, std::string params);

        /* Adds a module of the given |type| for the given |layer| */
        void add_module(Layer *layer, std::string type, std::string params);

        /* Reorganizes the model to place input and ouput layers in
         *   contiguous memory. */
        void sort_layers();

        // Driver string indicating type of driver
        std::string driver_string;

        // Total number of neurons
        int num_neurons;

        // Layers
        std::vector<Layer*> layers[IO_TYPE_SIZE];
        std::vector<Layer*> all_layers;

        // Connections
        std::vector<Connection*> connections;

        // Input and output modules
        std::vector<Module*> modules;

    private:
        /* Adds a given number of neurons.
         * This is called from add_layer() */
        void add_neurons(int count) {
            this->num_neurons += count;
        }
};

#endif
