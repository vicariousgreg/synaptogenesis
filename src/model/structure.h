#ifndef structure_h
#define structure_h

#include <vector>
#include <map>
#include <string>

#include "constants.h"
#include "model/layer.h"
#include "model/connection.h"
#include "io/module.h"

/* Represents a neural structure in a network model.
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
class Structure {
    public:
        Structure (std::string name);
        virtual ~Structure() { }

        static void connect(
            Structure *from_structure, std::string from_layer_name,
            Structure *to_structure, std::string to_layer_name,
            bool plastic, int delay, float max_weight, ConnectionType type,
            Opcode opcode, std::string params);


        /* Adds a layer to the environment with the given parameters */
        void add_layer(std::string name, int rows, int columns, std::string params);
        void add_layer_from_image(std::string name, std::string path, std::string params);

        /* Connects two layers, creating a weight matrix with the given
         *   parameters */
        Connection* connect_layers(std::string from_layer_name, std::string to_layer_name,
            bool plastic, int delay, float max_weight, ConnectionType type,
            Opcode opcode, std::string params);

        /* Connects to layers, sharing weights with another connection
         *   specified by |parent_id| */
        Connection* connect_layers_shared(
            std::string from_layer_name, std::string to_layer_name, Connection* parent);

        /* Uses expected sizes to create a new layer and connect it to the
         *   given layer.  Returns the id of the new layer. */
        Connection* connect_layers_expected(
            std::string from_layer_name, std::string to_layer_name, std::string new_layer_params,
            bool plastic, int delay, float max_weight,
            ConnectionType type, Opcode opcode, std::string params);

        /* Adds a module of the given |type| for the given |layer| */
        void add_module(std::string layer, std::string type, std::string params);

        // Structure name
        std::string name;

        // Layers
        std::vector<Layer*> layers;
        std::map<std::string, Layer*> layers_by_name;

        // Connections
        std::vector<Connection*> connections;

    private:
        Connection* connect_layers(
                Layer *from_layer, Layer *to_layer,
                bool plastic, int delay, float max_weight,
                ConnectionType type, Opcode opcode, std::string params);

        Connection* connect_layers(
                Layer *from_layer, Layer *to_layer,
                Connection *parent);

        Layer* find_layer(std::string name) {
            if (layers_by_name.find(name) != layers_by_name.end())
                return layers_by_name.find(name)->second;
            else
                return NULL;
        }
};

#endif
