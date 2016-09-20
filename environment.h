#ifndef environment_h
#define environment_h

#include <vector>
#include <ctime>
#include <cstdlib>

#include "layer.h"
#include "connectivity_matrix.h"

#define SPIKE_THRESH 30
#define EULER_RES 10
#define HISTORY_SIZE 8

using namespace std;

class NeuronParameters {
    public:
        NeuronParameters(float a, float b, float c, float d) :
                a(a), b(b), c(c), d(d) {}

        NeuronParameters copy() {
            return NeuronParameters(this->a, this->b, this->c, this->d);
        }

        float a;
        float b;
        float c;
        float d;
};

enum NeuronAttributes {
    CURRENT,
    VOLTAGE,
    RECOVERY,
    PARAMS,
    SIZE
};

class Environment {
    public:
        Environment () {
            this->num_neurons = 0;
            this->num_layers = 0;
            this->num_connections = 0;
            srand(time(NULL));
            //srand(0);
        }

        virtual ~Environment () {}

        void build();

        int add_layer(int size, int sign,
            float a, float b, float c, float d);
        int add_randomized_layer(int size, int sign);
        int connect_layers(int from_layer, int to_layer,
            bool plastic, float max_weight);
        int add_neuron(float a, float b, float c, float d);

        void inject_random_current(int layer_id, float max);
        void inject_current(int layer_id, float* input);

        int* get_spikes();
        float* get_currents();

        void cycle();
        void activate();
        void update_voltages();
        void timestep();
        void update_weights();

    //private:
        // Neurons
        int num_neurons;

        // Layers
        int num_layers;
        vector<Layer> layers;

        // Connection matrices.
        int num_connections;
        vector<ConnectivityMatrix> connections;

        // Parameter Vector
        vector<NeuronParameters> neuron_parameters;

        // Neuron Attribute Table
        void **nat;

        // Neuron Spikes
        int* spikes;
        int* recent_spikes;

#ifdef parallel
        int* local_spikes;
        float* local_currents;
#endif

    protected:
};

#endif
