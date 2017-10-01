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
 * Contains layers (nodes) and connections (edges) in the network graph.
 * Connections belong to the structure that holds the postsynaptic layer.
 *
 * Each structure is assigned a cluster in the driver that determines the
 *     sequence of events that happen to its state (ie whether state update
 *     or learning happens between layer activations or in a batch at the
 *     end of the timestep). The type of cluster is determined by the
 *     ClusterType passed into the constructor.
 * Note that not all neural models are supported by each cluster type.  This is
 *     checked for in Attributes::check_compatibility() when the cluster is
 *     constructed.
 */
class Structure {
    public:
        virtual ~Structure();

        /* Add or retrieve a layer to/from the structure */
        Layer* add_layer(const LayerConfig *layer_config);
        Layer* get_layer(std::string name, bool log_error=true) const;

        /* Connects two layers */
        static Connection* connect(
            Structure *from_structure,
            Structure *to_structure,
            const ConnectionConfig *conn_config);

        /* Getters */
        const StructureConfig* get_config() { return config; }
        const LayerList& get_layers() const { return layers; }
        const ConnectionList& get_connections() const { return connections; }

        // Checks whether this structure contains layers of a given neural_model
        bool contains(std::string neural_model) const;
        int get_num_neurons() const;
        std::string str() const { return "[Structure: " + name + "]"; }

        const std::string name;
        const ClusterType cluster_type;

    protected:
        friend class Network;

        Structure(StructureConfig* config);
        Structure(std::string name, ClusterType cluster_type=PARALLEL);

    private:
        Layer* add_layer_internal(const LayerConfig *layer_config);

        StructureConfig* const config;
        LayerList layers;
        ConnectionList connections;

        // Neural models used in this structure
        std::set<std::string> neural_model_flags;
};

typedef std::vector<Structure*> StructureList;

#endif
