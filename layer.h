#ifndef layer_h
#define layer_h

class Layer {
    public:
        Layer(int start_index, int size, int layer_index, int sign);

        virtual ~Layer () {}

        // Index of first neuron
        int start_index;

        // Size of layer
        int size;

        // Index of layer
        int layer_index;

        // Sign of layer (excitatory / inhibitory)
        int sign;

    private:
    protected:
};

#endif
