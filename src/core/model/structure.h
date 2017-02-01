#ifndef structure_h
#define structure_h

#include <map>
#include <string>

#include "util/constants.h"
#include "model/layer.h"
#include "model/connection.h"
#include "io/module/module.h"

/* Represents a neural structure in a network model.
 * Contains network graph data and parameters for layers/connections.
 *
 * Layers and connections can be created using a variety of functions.
 * In addition, input and output modules can be attached to layers using
 *    add_module(). These modules contain hooks for driving input to a layer
 *    and extracting and using output of a layer.
 */
class Structure {
    public:
        Structure (std::string name) : name(name) { }
        virtual ~Structure() { }

        /* Connects layers in two different structures */
        static Connection* connect(
            Structure *from_structure, std::string from_layer_name,
            Structure *to_structure, std::string to_layer_name,
            bool plastic, int delay, float max_weight, ConnectionType type,
            Opcode opcode, std::string params);


        /*******************************/
        /************ LAYERS ***********/
        /*******************************/
        void add_layer(std::string name, int rows, int columns, std::string params);
        void add_layer_from_image(std::string name, std::string path, std::string params);

        /*******************************/
        /********* CONNECTIONS *********/
        /*******************************/
        Connection* connect_layers(std::string from_layer_name, std::string to_layer_name,
            bool plastic, int delay, float max_weight, ConnectionType type,
            Opcode opcode, std::string params);

        Connection* connect_layers_shared(
            std::string from_layer_name, std::string to_layer_name, Connection* parent);

        Connection* connect_layers_expected(
            std::string from_layer_name, std::string to_layer_name, std::string new_layer_params,
            bool plastic, int delay, float max_weight,
            ConnectionType type, Opcode opcode, std::string params);

        Connection* connect_layers_matching(
            std::string from_layer_name, std::string to_layer_name, std::string new_layer_params,
            bool plastic, int delay, float max_weight,
            ConnectionType type, Opcode opcode, std::string params);


        /* Dendritic internal connection functions */
        DendriticNode *spawn_dendritic_node(std::string to_layer_name);

        Connection* connect_layers_internal(DendriticNode *node,
            std::string from_layer_name, std::string to_layer_name,
            bool plastic, int delay, float max_weight, ConnectionType type,
            Opcode opcode, std::string params);


        /* Adds a module of the given |type| for the given |layer| */
        void add_module(std::string layer, std::string type, std::string params);

        // Structure name
        const std::string name;

    private:
        friend class Model;

        /* Internal layer connection functions */
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

        // Layers
        LayerList layers;
        std::map<std::string, Layer*> layers_by_name;

        // Connections
        ConnectionList connections;
};

#endif
