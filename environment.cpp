#include <cstdlib>
#include <ctime>
#include <cstdio>

#include "environment.h"
#include "weight_matrix.h"
#include "operations.h"

Environment::Environment (Model model)
        : model(model) {
    srand(time(NULL));
    //srand(0);

    // Build the state
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
            throw "Failed to build environment!";
        }
    }
}


/******************************************************************************
 *************************** STATE INTERACTION ********************************
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
 ************************ TIMESTEP DYNAMICS ***********************************
 ******************************************************************************/

/*
 * Performs a timestep cycle.
 */
void Environment::timestep() {
    this->step_currents();
    this->step_voltages();
    this->step_spikes();
    this->step_weights();
}

/*
 * Performs activation during a timestep.
 * For each weight matrix, calculate sum of inputs for each neuron
 *   and add it to the current vector.
 */
void Environment::step_currents() {
    float* current = this->state.current;
    int* spikes = this->state.spikes;

    /* 2. Activation */
    // For each weight matrix...
    //   Update Currents using synaptic input
    //     current = operation ( current , dot ( spikes * weights ) )
    //
    // TODO: optimize order, create batches of parallelizable computations,
    //       and move cuda barriers around batches
    for (int cid = 0 ; cid < this->model.num_connections; ++cid) {
        update_currents(this->model.connections[cid],
            spikes, current, this->model.num_neurons);
    }
}


/*
 * Perform voltage update according to input currents using Izhikevich
 *   with Euler's method.
 */
void Environment::step_voltages() {
    /* 3. Voltage Updates */
    update_voltages(
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
void Environment::step_spikes() {
    /* 4. Timestep */
    update_spikes(
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
void Environment::step_weights() {
    /* 5. Update weights */
}
