#ifndef layer_h
#define layer_h

#include <vector>
#include <set>
#include <string>

#include "model/layer_config.h"
#include "model/dendritic_node.h"
#include "util/constants.h"

class Structure;
class Connection;
typedef std::vector<Connection*> ConnectionList;
class ModuleConfig;

/* Represents a two dimensional layer of neurons.
 * Layers can be constructed and connected into networks using the Structure class.
 * Layers contain a boatload of information.
 */
class Layer {
    public:
        virtual ~Layer();

        /* Constant getters */
        const LayerConfig* get_config() const;
        IOTypeMask get_type() const;
        bool is_input() const;
        bool is_output() const;
        bool is_expected() const;

        std::string get_parameter(
            std::string key, std::string default_val) const;
        const std::vector<ModuleConfig*> get_module_configs() const;

        const ConnectionList& get_input_connections() const;
        const ConnectionList& get_output_connections() const;

        // Layer name
        const std::string name;

        // Layer ID
        const int id;

        // Neural model
        const std::string neural_model;

        // Housing structure
        Structure* const structure;

        // Layer rows, columns, and total size
        const int rows, columns, size;

        // Noise parameters
        const float noise_mean;
        const float noise_std_dev;

        // Plasticity flag
        const bool plastic;

        // Global flag
        const bool global;

        // Root node of dendritic tree
        DendriticNode* const dendritic_root;

        // Gets the maximum delay for all outgoing connections
        int get_max_delay() const;

    private:
        friend class Structure;
        friend class Connection;

        // Global counter for ID assignment
        static int count;

        Layer(Structure *structure, LayerConfig *config);

        // Methods for adding connections and modules
        void add_input_connection(Connection* connection);
        void add_output_connection(Connection* connection);
        void add_to_root(Connection* connection);
        void add_module(ModuleConfig *config);

        // Layer IO type mask
        IOTypeMask type;

        // Modules
        std::vector<ModuleConfig*> module_configs;

        // Input and output connections
        ConnectionList input_connections;
        ConnectionList output_connections;

        // Layer config
        LayerConfig* const config;
};

typedef std::vector<Layer*> LayerList;

#endif
