#include <cstdlib>
#include <ctime>
#include <cstdio>

#include "environment.h"
#include "weight_matrix.h"
#include "operations.h"
#include "parallel.h"

Environment::Environment (Model model)
        : model(model) {
    srand(time(NULL));
    //srand(0);
}

/******************************************************************************
 ************************* GETTER / SETTER ************************************
 ******************************************************************************/

int* Environment::get_spikes() {
    return this->state.get_spikes();
}

float* Environment::get_current() {
    return this->state.get_current();
}

void Environment::inject_current(int layer_id, float* input) {
    int offset = this->model.layers[layer_id].index;
    int size = this->model.layers[layer_id].size;
    this->state.set_current(offset, size, input);
}

void Environment::inject_random_current(int layer_id, float max) {
    int offset = this->model.layers[layer_id].index;
    int size = this->model.layers[layer_id].size;
    this->state.randomize_current(offset, size, max);
}

void Environment::clear_current(int layer_id) {
    int offset = this->model.layers[layer_id].index;
    int size = this->model.layers[layer_id].size;
    this->state.clear_current(offset, size);
}


/******************************************************************************
 ********************** INITIALIZATION FUNCTIONS ******************************
 ******************************************************************************/

/*
 * Builds the environment.
 * During dynamic construction, instantiation is lazy.
 * Neuron parameters are tracked in a vector, but the neuron attributes table
 *   is not initialized until this function is called.
 */
void Environment::build() {
    // Build the state.
    try {
        this->state.build(this->model);
    } catch (const char* msg) {
        printf("%s\n", msg);
        printf("Failed to build environment state!\n");
        throw "Failed to build environment!";
    }

    // Build weight matrices
    for (int i = 0 ; i < this->model.num_connections ; ++i) {
        WeightMatrix &conn = this->model.connections[i];
        try {
            conn.build();
        } catch (const char* msg) {
            printf("%s\n", msg);
            printf("Failed to initialize %d x %d matrix!\n",
                conn.from_layer.size, conn.to_layer.size);
            throw "Failed to environment!";
        }
    }
}


/******************************************************************************
 ************************ TIMESTEP DYNAMICS ***********************************
 ******************************************************************************/

/*
 * Performs a timestep cycle.
 */
void Environment::timestep() {
    this->activate();
    this->update_voltages();
    this->cycle_spikes();
    this->update_weights();
}

/*
 * Performs activation during a timestep.
 * For each weight matrix, calculate sum of inputs for each neuron
 *   and add it to the current vector.
 */
void Environment::activate() {
    float* current = this->state.current;
    int* spikes = this->state.recent_spikes;

    /* 2. Activation */
    // For each weight matrix...
    //   Update Currents using synaptic input
    //     current += sign * dot ( spikes * weights )
    //
    // TODO: optimize order, create batches of parallelizable computations,
    //       and move cuda barriers around batches
    for (int cid = 0 ; cid < this->model.num_connections; ++cid) {
        activate_conn(this->model.connections[cid], spikes, current);
    }
}


/*
 * Perform voltage update according to input currents using Izhikevich
 *   with Euler's method.
 */
void Environment::update_voltages() {
    /* 3. Voltage Updates */
    izhikevich(
        this->state.voltage,
        this->state.recovery,
        this->state.current,
        this->state.neuron_parameters,
        this->model.num_neurons);
}

/*
 * Perform timestep cycling.
 * Fills the spike buffer based on voltages and the SPIKE_THRESH.
 * Increments the ages of last spikes, and resets recovery if spiked.
 */
void Environment::cycle_spikes() {
    /* 4. Timestep */
    calc_spikes(
        this->state.spikes,
        this->state.voltage,
        this->state.recovery,
        this->state.neuron_parameters,
        this->model.num_neurons);
}

/**
 * Updates weights.
 * TODO: implement.
 */
void Environment::update_weights() {
    /* 5. Update weights */
}
