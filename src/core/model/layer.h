#ifndef layer_h
#define layer_h

#include <vector>
#include <string>

#include "model/layer_config.h"
#include "model/dendritic_node.h"
#include "util/constants.h"

class Structure;
class Connection;
typedef std::vector<Connection*> ConnectionList;
class Module;
typedef std::vector<Module*> ModuleList;

/* Represents a two dimensional layer of neurons.
 * Layers can be constructed and connected into networks using the Structure class.
 * Layers contain a boatload of information.
 */
class Layer {
    public:
        virtual ~Layer();

        /* Constant getters */
        IOTypeMask get_type() const;
        bool is_input() const;
        bool is_output() const;
        bool is_expected() const;

        Module* get_input_module() const;
        Module* get_expected_module() const;
        const ModuleList get_output_modules() const;

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

        // Config
        LayerConfig* const config;

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
        void add_module(Module *module);

        // Layer IO type mask
        IOTypeMask type;

        // Modules
        Module* input_module;
        Module* expected_module;
        ModuleList output_modules;

        // Input and output connections
        ConnectionList input_connections;
        ConnectionList output_connections;
};

typedef std::vector<Layer*> LayerList;

#endif
