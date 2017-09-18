#ifndef layer_h
#define layer_h

#include <vector>
#include <set>
#include <string>

#include "network/layer_config.h"
#include "network/dendritic_node.h"
#include "util/constants.h"

class Structure;
class Connection;
typedef std::vector<Connection*> ConnectionList;

/* Represents a two dimensional layer of neurons.
 * Layers can be constructed and connected into networks using the Structure class.
 * Layers contain a boatload of information.
 */
class Layer {
    public:
        virtual ~Layer();

        /* Constant getters */
        const LayerConfig* get_config() const;

        std::string get_parameter(
            std::string key, std::string default_val) const;

        const ConnectionList& get_input_connections() const;
        const ConnectionList& get_output_connections() const;

        // Layer name
        const std::string name;

        // Layer ID
        const size_t id;

        // Neural model
        const std::string neural_model;

        // Housing structure
        Structure* const structure;

        // Layer rows, columns, and total size
        const int rows, columns, size;

        // Plasticity flag
        const bool plastic;

        // Global flag
        const bool global;

        // Root node of dendritic tree
        DendriticNode* const dendritic_root;

        // Get a list of the dendritic nodes
        DendriticNodeList get_dendritic_nodes() const;

        // Get dendritic node by name
        DendriticNode* get_dendritic_node(std::string name,
            bool log_error=false) const;

        // Gets the maximum delay for all outgoing connections
        int get_max_delay() const;

        // Gets the total number of incoming weights
        int get_num_weights() const;

        std::string str() const;

    private:
        friend class Network;
        friend class Structure;
        friend class Connection;
        friend class DendriticNode;

        Layer(Structure *structure, LayerConfig *config);

        // Methods for adding connections
        void add_input_connection(Connection* connection);
        void add_output_connection(Connection* connection);
        void add_to_root(Connection* connection);

        // Layer IO type mask
        IOTypeMask type;

        // Input and output connections
        ConnectionList input_connections;
        ConnectionList output_connections;

        // Layer config
        LayerConfig* const config;
};

typedef std::vector<Layer*> LayerList;

/* Counts the number of connections in a layer list */
inline int get_num_connections(LayerList& layers) {
    int num_connections = 0;
    for (auto& layer : layers)
        num_connections += layer->get_input_connections().size();
}

#endif
