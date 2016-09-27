#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "model.h"
#include "state.h"
#include "izhikevich_state.h"
#include "izhikevich_driver.h"
#include "rate_encoding_state.h"
#include "tools.h"

static Timer timer = Timer();

Model* build_model() {
    /* Construct the model */
    Model *model = new Model();
    int size = 800 * 1;

    int pos = model->add_layer(size, "random positive");
    int neg = model->add_layer(size / 4, "random negative");
    model->connect_layers(pos, pos, true, 0, .5, FULLY_CONNECTED, ADD);
    model->connect_layers(pos, neg, true, 0, .5, FULLY_CONNECTED, ADD);
    model->connect_layers(neg, pos, true, 0, 1, FULLY_CONNECTED, SUB);
    model->connect_layers(neg, neg, true, 0, 1, FULLY_CONNECTED, SUB);

    return model;
}

IzhikevichState* build_izhikevich_state(Model* model) {
    /* Construct the state */
    IzhikevichState *state = new IzhikevichState();
    state->build(model);
    return state;
}

RateEncodingState* build_rate_encoding_state(Model* model) {
    /* Construct the state */
    RateEncodingState *state = new RateEncodingState();
    state->build(model);
    return state;
}

/* Prints a line for a timestep containing markers for neuron spikes.
 * If a neuron spikes, an asterisk will be printed.  Otherwise, a space */
void print_spikes(IzhikevichState *state) {
    int* spikes = (int*)state->get_output();
    for (int nid = 0 ; nid < state->model->num_neurons ; ++nid) {
        char c = (spikes[nid] % 2) ? '*' : ' ';
        std::cout << c;
    }
    std::cout << "|\n";
}

void print_outputs(RateEncodingState *state) {
    float* output = (float*)state->get_output();
    for (int nid = 0 ; nid < state->model->num_neurons ; ++nid) {
        std::cout << output[nid] << " ";
    }
    std::cout << "|\n";
}

/* Prints a line for a timestep containing neuron currents */
void print_currents(IzhikevichState *state) {
    float* currents = state->get_input();
    for (int nid = 0 ; nid < state->model->num_neurons ; ++nid) {
        std::cout << currents[nid] << " " ;
    }
    std::cout << "|\n";
}

void test_izhikevich(Model* model) {
    // Start timer
    timer.start();

    IzhikevichDriver driver;
    driver.build(model);
    IzhikevichState *state = build_izhikevich_state(model);
    printf("Built state.\n");
    timer.stop("Initialization");

    timer.start();
    int iterations = 50;
    for (int i = 0 ; i < iterations ; ++i) {
        driver.state->randomize_input(0, 5);
        driver.state->randomize_input(1, 2);
        driver.timestep();
        //print_currents(driver.state);
        print_spikes((IzhikevichState*)driver.state);
    }

    float time = timer.stop("Total time");
    printf("Time averaged over %d iterations: %f\n", iterations, time/iterations);
}

void test_rate_encoding(Model* model) {
    // Start timer
    timer.start();

    RateEncodingState *state = build_rate_encoding_state(model);
    printf("Built state.\n");
    timer.stop("Initialization");

    timer.start();
    int iterations = 50;
    for (int i = 0 ; i < iterations ; ++i) {
        state->randomize_input(0, 0.5);
        state->randomize_input(1, 0.2);
        state->timestep();
        print_outputs(state);
    }

    float time = timer.stop("Total time");
    printf("Time averaged over %d iterations: %f\n", iterations, time/iterations);
}

int main(void) {
    // Seed random number generator
    srand(time(NULL));

    try {
        // Start timer
        timer.start();

        Model *model = build_model();
        printf("Built model.\n");
        printf("  - neurons     : %10d\n", model->num_neurons);
        printf("  - layers      : %10d\n", model->num_layers);
        printf("  - connections : %10d\n", model->num_connections);

        //test_rate_encoding(model);
        test_izhikevich(model);

    } catch (const char* msg) {
        printf("\n\nERROR: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }

    return 0;
}
