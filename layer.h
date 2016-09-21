#ifndef layer_h
#define layer_h

class Layer {
    public:
        /* Constructor.
         * |start_index| identifies the first neuron in the layer.
         * |size| indicates the size of the layer.
         * |sign| specifies the output sign of neural activity.
         * TODO: replace sign with function
         */
        Layer(int start_index, int size, int sign);

        // Index of first neuron
        int start_index;

        // Size of layer
        int size;

        // Sign of layer (excitatory / inhibitory)
        int sign;
};

#endif
