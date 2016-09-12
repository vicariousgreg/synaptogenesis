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

        int a;
        int b;
        int c;
        int d;
};

class Environment {
    public:
        Environment () {
            this->num_neurons = 0;
            this->num_layers = 0;
            this->num_connections = 0;
            srand(time(NULL));
        }

        virtual ~Environment () {}

        int add_layer(int size, int sign, double a, double b, double c, double d);

        int connect_layers(int from_layer, int to_layer, bool plastic);

        int add_neuron(double a, double b, double c, double d);

        void cycle();

    //private:
        // Neurons
        int num_neurons;

        // Layers
        int num_layers;
        vector<Layer> layers;

        // Connection matrices.
        int num_connections;
        vector<ConnectivityMatrix> connections;

        // Age Vector
        vector<int> ages;

        // Spike Vector
        vector<bool> spikes;

        // Voltage Vector
        vector<double> voltages;

        // Current Vector
        vector<double> currents;

        // Recovery Vector
        vector<double> recoveries;

        // Parameter Vector
        vector<NeuronParameters> neuron_parameters;

    protected:
};

#endif
