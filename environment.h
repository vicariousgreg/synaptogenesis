#ifndef environment_h
#define environment_h

#include <vector>
#include <ctime>
#include <cstdlib>

#include "layer.h"
#include "connectivity_matrix.h"

using namespace std;

class NeuronParameters {
    public:
        NeuronParameters(double a, double b, double c, double d) :
                a(a), b(b), c(c), d(d) {}

        double a;
        double b;
        double c;
        double d;
};

enum NeuronAttributes {
    CURRENT,
    VOLTAGE,
    RECOVERY,
    SPIKE,
    AGE,
    PARAMS_A,
    PARAMS_B,
    PARAMS_C,
    PARAMS_D,
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
            double a, double b, double c, double d);
        int add_randomized_layer(int size, int sign);
        int connect_layers(int from_layer, int to_layer,
            bool plastic, double max_weight);
        int add_neuron(double a, double b, double c, double d);

        void set_random_currents(int layer_id, double max);

        void cycle();
        void gather_input();
        void activate();
        void update_voltages();
        void timestep();
        void update_weights();
        void report_output();

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

    protected:
};

#endif
