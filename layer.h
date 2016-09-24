#ifndef layer_h
#define layer_h

class WeightMatrix;

class Layer {
    public:
        /* Constructor.
         * |start_index| identifies the first neuron in the layer.
         * |size| indicates the size of the layer.
         */
        Layer(int start_index, int size);

        // Index of first neuron
        int index;

        // Size of layer
        int size;
};

#endif
