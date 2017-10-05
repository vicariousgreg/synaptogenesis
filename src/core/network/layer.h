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

        /* Getters */
        const LayerConfig* get_config() const { return config; }

        const ConnectionList& get_input_connections() const;
        const ConnectionList& get_output_connections() const;

        /* Checks if layer is an input or output layer within its structure */
        bool is_structure_input() const;
        bool is_structure_output() const;

        DendriticNodeList get_dendritic_nodes() const;
        DendriticNode* get_dendritic_root() const { return dendritic_root; }
        DendriticNode* get_dendritic_node(std::string name,
            bool log_error=false) const;

        std::string get_parameter( std::string key, std::string dev_val) const;

        /* Gets the maximum delay for all outgoing connections */
        int get_max_delay() const;

        /* Gets the total number of incoming weights */
        int get_num_weights() const;

        std::string str() const;

        const LayerConfig * const config;
        const std::string name;
        const size_t id;
        const std::string neural_model;
        Structure* const structure;
        const int rows, columns, size;
        const bool plastic;
        const bool global;

    protected:
        friend class Structure;

        Layer(Structure *structure, const LayerConfig *config);

    protected:
        friend class Connection;

        void add_input_connection(Connection* connection);
        void add_output_connection(Connection* connection);

    private:
        void add_dendrites(std::string parent_name,
            const ConfigArray& dendrites);

        DendriticNode* const dendritic_root;
        ConnectionList input_connections;
        ConnectionList output_connections;
};

typedef std::vector<Layer*> LayerList;

/* Counts the number of connections in a layer list */
int get_num_connections(const LayerList& layers);

/* Checks if layers in a list are the same size */
bool check_equal_sizes(const LayerList& layers);

#endif
