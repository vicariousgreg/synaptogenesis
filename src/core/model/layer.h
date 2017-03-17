#ifndef layer_h
#define layer_h

#include <vector>
#include <string>

#include "model/dendritic_node.h"
#include "model/layer_config.h"
#include "util/constants.h"

class Structure;
class Connection;
class Module;
typedef std::vector<Module*> ModuleList;

/* Represents a two dimensional layer of neurons.
 * Layers can be constructed and connected into networks using the Model class.
 *
 * Layers contain:
 *   - unique identifier
 *   - starting index in the neural arrays
 *   - size information
 *   - parameters for matrix initialization
 *
 */
class Layer {
    public:
        virtual ~Layer() {
            delete this->dendritic_root;
        }

        /* Constant getters */
        IOTypeMask get_type() const { return type; }
        bool is_input() const { return type & INPUT; }
        bool is_output() const { return type & OUTPUT; }
        bool is_error() const { return type & ERROR; }

        Module* get_input_module() const { return input_module; }
        const ModuleList get_output_modules() const { return output_modules; }

        const ConnectionList& get_input_connections() const { return input_connections; }
        const ConnectionList& get_output_connections() const { return output_connections; }

        // Layer name
        const std::string name;

        // Layer ID
        const int id;

        // Neural model
        const NeuralModel neural_model;

        // Housing structure
        Structure* const structure;

        // Layer rows, columns, and total size
        const int rows, columns, size;

        // Parameters for initializing neural properties
        const std::string params;

        // Noise parameter
        const float noise;

        // Plasticity flag
        bool plastic;

        // Root node of dendritic tree
        DendriticNode* const dendritic_root;

    private:
        friend class Model;
        friend class Structure;
        friend class Connection;

        static int count;

        Layer(Structure *structure, LayerConfig config);

        void add_input_connection(Connection* connection);
        void add_output_connection(Connection* connection);
        void add_to_root(Connection* connection);
        void add_module(Module *module);

        // Layer type (input, output, input/output, internal)
        IOTypeMask type;

        // Modules
        Module* input_module;
        Module* error_module;
        ModuleList output_modules;

        // Input and output connections
        ConnectionList input_connections;
        ConnectionList output_connections;
};

typedef std::vector<Layer*> LayerList;

#endif
