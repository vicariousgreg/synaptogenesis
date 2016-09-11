#include <vector>
#include "layer.h"

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
        }

        virtual ~Environment () {}

        int add_layer(int size, int sign, double a, double b, double c, double d);

        int add_neuron(double a, double b, double c, double d);

        void cycle();

    //private:
        // Count of neurons
        int num_neurons;

        // Count of layers
        int num_layers;

        // Layer Vector
        vector<Layer> layers;

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
