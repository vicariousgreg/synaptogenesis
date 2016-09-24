#ifndef layer_h
#define layer_h

#include <vector>

class WeightMatrix;

class Layer {
    public:
        /* Constructor.
         * |start_index| identifies the first neuron in the layer.
         * |size| indicates the size of the layer.
         * |sign| specifies the output sign of neural activity.
         * TODO: replace sign with function
         */
        Layer(int start_index, int size, int sign);

        void add_incoming_connection(WeightMatrix &matrix);

        // Index of first neuron
        int index;

        // Size of layer
        int size;

        // Sign of layer (excitatory / inhibitory)
        int sign;

        // Incoming weight matrices
        std::vector<WeightMatrix*> incoming_connections;
};

#endif
