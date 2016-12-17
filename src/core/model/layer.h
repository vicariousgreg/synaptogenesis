#ifndef layer_h
#define layer_h

#include <vector>
#include <string>

#include "util/constants.h"

class Connection;
class Module;

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
        // Layer name
        std::string name;

        // Layer id
        int id;

        // Start index
        int start_index;

        // Layer rows, columns, and total size
        int rows, columns, size;

        // Parameters for initializing neural properties
        std::string params;

        // Layer type (input, output, input/output, internal)
        IOType type;

        // Indices relative to input/output, if relevant
        int input_index, output_index;

        // Input and output connections
        std::vector<Connection*> input_connections;
        std::vector<Connection*> output_connections;

        Module *input_module;
        std::vector<Module*> output_modules;

    private:
        friend class Model;
        friend class Structure;
        friend class Connection;

        Layer(std::string name, int rows, int columns, std::string params);

        void add_input_connection(Connection* connection) {
            this->input_connections.push_back(connection);
        }

        void add_output_connection(Connection* connection) {
            this->output_connections.push_back(connection);
        }

        void add_module(Module *module);
};

#endif
