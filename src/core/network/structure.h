#ifndef structure_h
#define structure_h

#include <map>
#include <set>
#include <vector>
#include <string>

#include "network/layer.h"
#include "network/connection.h"
#include "network/structure_config.h"
#include "util/constants.h"
#include "util/property_config.h"

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
 */
class Structure {
    public:
        Structure(StructureConfig* config);
        Structure(std::string name, ClusterType cluster_type=PARALLEL);
        virtual ~Structure();

        /* Checks whether this structure contains a layer
         *     of the given neural_model */
        bool contains(std::string neural_model) {
            return neural_model_flags.count(neural_model);
        }

        /* Gets the total neuron count */
        int get_num_neurons() const {
            int num_neurons = 0;
            for (auto layer : layers)
                num_neurons += layer->size;
            return num_neurons;
        }


        /*******************************/
        /************ LAYERS ***********/
        /*******************************/
        const LayerList& get_layers() const { return layers; }
        Layer* add_layer(LayerConfig *layer_config);

        /*******************************/
        /********* CONNECTIONS *********/
        /*******************************/
        const ConnectionList& get_connections() const { return connections; }

        /* Connects layers in two different structures */
        static Connection* connect(
            Structure *from_structure, std::string from_layer_name,
            Structure *to_structure, std::string to_layer_name,
            ConnectionConfig *conn_config,
            std::string node="root",
            std::string name="");

        // Gets the name of the internal node above the
        //   connection's leaf dendritic node
        std::string get_parent_node_name(Connection *conn) const;

        // Structure name
        const std::string name;

        // Stream type for iteration computation order
        const ClusterType cluster_type;

        std::string str() const { return "[Structure: " + name + "]"; }

        /* Find a layer
         * If not found, logs an error or returns nullptr */
        Layer* get_layer(std::string name, bool log_error=true) const;

        // Structure config
        StructureConfig* const config;

    protected:
        Layer* add_layer_internal(LayerConfig *layer_config);

        /* Internal layer connection function */
        Connection* connect_layers(
                Layer *from_layer, Layer *to_layer,
                ConnectionConfig *conn_config,
                std::string node="root",
                std::string name="");

        // Layers
        LayerList layers;

        // Connections
        ConnectionList connections;

        // Neural models used in this structure
        std::set<std::string> neural_model_flags;
};

typedef std::vector<Structure*> StructureList;

#endif
