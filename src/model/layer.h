#ifndef layer_h
#define layer_h

#include <string>

#include "constants.h"

/* Forward declarations for input and output modules */
class Input;
class Output;

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

    private:
        friend class Model;

        /* Constructor.
         * |start_index| identifies the first neuron in the layer.
         * |size| indicates the size of the layer.
         */
        Layer(int layer_id, int start_index, int rows, int columns) :
                id(layer_id),
                index(start_index),
                rows(rows),
                columns(columns),
                size(rows * columns) {}
};

#endif
