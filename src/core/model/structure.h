#ifndef structure_h
#define structure_h

#include <map>
#include <set>
#include <vector>
#include <string>

#include "util/constants.h"
#include "model/layer.h"
#include "model/connection.h"
#include "model/property_config.h"
#include "io/module/module.h"

/* Represents a neural structure in a network model.
 * Contains network graph data and parameters for layers/connections.
 *
 * Each structure is assigned a cluster in the driver that determines the
 *     sequence of events that happen to its state (ie whether learning happens
 *     netween layer activations or in a batch at the end of the timestep).
 *     The type of cluster is determined by the ClusterType passed into the
 *     constructor.
 * Note that not all neural models are supported by each cluster type.  This is
 *     checked for in Attributes::check_compatibility() when the cluster is
 *     constructed.
 *
 * Layers and connections can be created using a variety of functions.
 * In addition, input and output modules can be attached to layers using
 *    add_module(). These modules contain hooks for driving input to a layer
 *    and extracting and using output of a layer.
 */
class Structure {
    public:
        Structure(std::string name, ClusterType cluster_type=PARALLEL);
        virtual ~Structure();

        /* Checks whether this structure contains a layer
         *     of the given neural_model */
        bool contains(std::string neural_model) {
            return neural_model_flags.count(neural_model);
        }

        /* Gets the total neuron count */
        int get_num_neurons() const { return total_neurons; }


        /*******************************/
        /************ LAYERS ***********/
        /*******************************/
        const LayerList& get_layers() const { return layers; }
        void add_layer(LayerConfig *config);
        void add_layer_from_image(std::string path, LayerConfig *config);

        /*******************************/
        /********* CONNECTIONS *********/
        /*******************************/
        const ConnectionList& get_connections() const { return connections; }

        /* Connects layers in two different structures */
        static Connection* connect(
            Structure *from_structure, std::string from_layer_name,
            Structure *to_structure, std::string to_layer_name,
            ConnectionConfig *config);

        Connection* connect_layers(std::string from_layer_name,
            std::string to_layer_name, ConnectionConfig *config);

        Connection* connect_layers_expected(
            std::string from_layer_name,
            LayerConfig *layer_config, ConnectionConfig *conn_config);

        Connection* connect_layers_matching(
            std::string from_layer_name,
            LayerConfig *layer_config, ConnectionConfig *conn_config);

        /*****************************/
        /********* DENDRITES *********/
        /*****************************/
        DendriticNode *get_dendritic_root(std::string to_layer_name);
        DendriticNode *spawn_dendritic_node(std::string to_layer_name);

        Connection* connect_layers_internal(
            DendriticNode *node, std::string from_layer_name,
            ConnectionConfig *config);


        /* Adds a module of the given |config| for the given |layer| */
        void add_module(std::string layer_name, ModuleConfig *config);
        void add_module_all(ModuleConfig *config);

        // Structure name
        const std::string name;

        // Stream type for iteration computation order
        const ClusterType cluster_type;

    private:
        /* Internal layer connection functions */
        Connection* connect_layers(
                Layer *from_layer, Layer *to_layer,
                ConnectionConfig *config);

        /* Find a layer
         * If not found, logs an error or returns nullptr */
        Layer* find_layer(std::string name, bool log_error=true);

        // Layers
        LayerList layers;
        std::map<std::string, Layer*> layers_by_name;

        // Connections
        ConnectionList connections;

        // Number of neurons
        int total_neurons;

        std::set<std::string> neural_model_flags;
};

typedef std::vector<Structure*> StructureList;

#endif
