#ifndef layer_h
#define layer_h

#include <vector>
#include <string>

#include "constants.h"

class Connection;

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
        // Layer ID and start index
        int id, index;

        // Layer rows, columns, and total size
        int rows, columns, size;

        // Parameters for initializing neural properties
        std::string params;

        // Flags for whether layer has input or output modules
        bool has_input_module, has_output_module;

        // Input and output connections
        std::vector<Connection*> input_connections;
        std::vector<Connection*> output_connections;

    private:
        friend class Model;

        /* Constructor.
         * |start_index| identifies the first neuron in the layer.
         * |size| indicates the size of the layer.
         */
        Layer(int layer_id, int start_index, int rows, int columns, std::string params) :
                id(layer_id),
                index(start_index),
                rows(rows),
                columns(columns),
                size(rows * columns),
                params(params),
                has_input_module(false),
                has_output_module(false) {}

        void add_input_connection(Connection* connection) {
            this->input_connections.push_back(connection);
        }

        void add_output_connection(Connection* connection) {
            this->output_connections.push_back(connection);
        }
};

#endif
