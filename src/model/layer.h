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

        // Layer type (input, output, input/output, internal)
        LayerType type;

        // Indices relative to input/output, if relevant
        int input_index, output_index;

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
                type(INTERNAL),
                input_index(0),
                output_index(0) {}

        void add_input_connection(Connection* connection) {
            this->input_connections.push_back(connection);
        }

        void add_output_connection(Connection* connection) {
            this->output_connections.push_back(connection);
        }

        void add_input_module() {
            if (this->type == OUTPUT)
                this->type = INPUT_OUTPUT;
            else if (this->type == INPUT or this->type == INPUT_OUTPUT)
                throw "Layer cannot have more than one input module!";
            else
                this->type = INPUT;
        }

        void add_output_module() {
            if (this->type == INPUT)
                this->type = INPUT_OUTPUT;
            else if (this->type == OUTPUT or this->type == INPUT_OUTPUT)
                throw "Layer cannot have more than one output module!";
            else
                this->type = OUTPUT;
        }
};

#endif
