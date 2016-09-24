#ifndef layer_h
#define layer_h

#include <vector>

class WeightMatrix;

class Layer {
    public:
        /* Constructor.
         * |start_index| identifies the first neuron in the layer.
         * |size| indicates the size of the layer.
         */
        Layer(int start_index, int size);

        void add_incoming_connection(WeightMatrix* matrix);

        // Index of first neuron
        int index;

        // Size of layer
        int size;

        // Incoming weight matrices
        std::vector<WeightMatrix*> incoming_connections;
};

#endif
