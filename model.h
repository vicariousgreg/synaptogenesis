#ifndef model_h
#define model_h

#include <vector>

#include "neuron_parameters.h"
#include "layer.h"
#include "weight_matrix.h"

class Model {
    public:
        Model ();

        /* Adds a layer to the environment with the given parameters */
        int add_layer(int size, int sign,
            float a, float b, float c, float d);

        /* Adds a randomized layer to the environment */
        int add_randomized_layer(int size, int sign);

        /* Connects two layers, creating a weight matrix with the given 
         *   parameters */
        int connect_layers(int from_layer, int to_layer,
            bool plastic, int delay, float max_weight, MatrixType type);

        // Neurons
        int num_neurons;

    private:
        friend class Environment;
        friend class State;

        /* Adds a single neuron.
         * This is called from add_layer() */
        int add_neuron(float a, float b, float c, float d);

        // Layers
        int num_layers;
        std::vector<Layer> layers;

        // Connection matrices.
        int num_connections;
        std::vector<WeightMatrix> connections;

        // Parameter Vector
        std::vector<NeuronParameters> neuron_parameters;
};

#endif
