#ifndef environment_h
#define environment_h

#include <vector>
#include <ctime>
#include <cstdlib>

#include "layer.h"
#include "weight_matrix.h"

#define SPIKE_THRESH 30
#define EULER_RES 10
#define HISTORY_SIZE 8

using namespace std;

/* Neuron parameters class.
 * Contains a,b,c,d parameters for Izhikevich model */
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

/* Enumeration for neuron attributes for memory layout purposes */
enum NeuronAttributes {
    CURRENT,
    VOLTAGE,
    RECOVERY,
    PARAMS,
    SIZE
};

class Environment {
    public:
        Environment ();

        virtual ~Environment () {}

        /* Builds the environment.
         * Allocates memory for various arrays of data.
         * Calls build() on all weight matrices. */
        void build();

        /* Adds a layer to the environment with the given parameters */
        int add_layer(int size, int sign,
            float a, float b, float c, float d);

        /* Adds a randomized layer to the environment */
        int add_randomized_layer(int size, int sign);

        /* Connects two layers, creating a weight matrix with the given 
         *   parameters */
        int connect_layers(int from_layer, int to_layer,
            bool plastic, float max_weight);

        /* Injects current from the given pointer to the given layer */
        void inject_current(int layer_id, float* input);

        /* Injects randomized current to the given layer.
         * Input is bounded by |max| */
        void inject_random_current(int layer_id, float max);

        /* Zeroes current to the given layer */
        void clear_current(int layer_id);

        /* Returns a pointer to an array containing the most recent integer
         *   of neuron spikes.
         * If parallel, this will copy the values from the device. */
        int* get_spikes();

        /* Returns a pointer to an array containing the neuron currents.
         * If parallel, this will copy the values from the device. */
        float* get_currents();

        /* Cycles the environment:
         * 1. Activate neural connections, which updates currents.
         * 2. Update neuron voltages from currents.
         * 3. Timestep the spikes.
         * 4. Update connection weights for plastic matrices */
        void cycle();

        /* Activates neural connections, triggering updates of currents */
        void activate();

        /* Updates neuron voltages from the currents using the Izhikevich model */
        void update_voltages();

        /* Timesteps the spikes, shifting spike bit vectors */
        void timestep();

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDB variant Hebbian learning */
        void update_weights();

        // Neurons
        int num_neurons;

    private:
        /* Adds a single neuron.
         * This is called from add_layer() */
        int add_neuron(float a, float b, float c, float d);

        // Layers
        int num_layers;
        vector<Layer> layers;

        // Connection matrices.
        int num_connections;
        vector<WeightMatrix> connections;

        // Parameter Vector
        vector<NeuronParameters> neuron_parameters;

        // Neuron Attribute Table
        void **nat;

        // Neuron Spikes
        // Recent points to the most recent integers for spike bit vectors
        int* spikes;
        int* recent_spikes;

#ifdef parallel
        // Locations to store local copies of spikes and currents.
        // When accessed, these values will be copied here from the device.
        int* local_spikes;
        float* local_currents;
#endif
};

#endif
