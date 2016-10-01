#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model.h"
#include "state.h"
#include "driver.h"
#include "tools.h"

static Timer timer = Timer();

Model* build_model(std::string driver_name, bool verbose) {
    /* Construct the model */
    Model *model = new Model(driver_name);

    int pos = model->add_layer(50, 50, "default");
    model->add_input(pos, "image", "bird-head-small.jpg");
    model->add_output(pos, "print_spike", "");

    /*
    int size = 800 * 1;
    int pos = model->add_layer(50, 50, "random positive");
    int neg = model->add_layer(1, size / 4, "random negative");
    model->connect_layers(pos, pos, true, 0, .5, FULLY_CONNECTED, ADD);
    model->connect_layers(pos, neg, true, 0, .5, FULLY_CONNECTED, ADD);
    model->connect_layers(neg, pos, true, 0, 1, FULLY_CONNECTED, SUB);
    model->connect_layers(neg, neg, true, 0, 1, FULLY_CONNECTED, SUB);
    model->add_input(pos, "random", "5");
    model->add_input(neg, "random", "2");
    */

    if (verbose) printf("Built model.\n");
    if (verbose) printf("  - neurons     : %10d\n", model->num_neurons);
    if (verbose) printf("  - layers      : %10d\n", model->num_layers);
    if (verbose) printf("  - connections : %10d\n", model->num_connections);


    return model;
}

int run_simulation(Model *model, int iterations, bool verbose) {
    // Seed random number generator
    srand(time(NULL));

    try {
        // Start timer
        timer.start();

        Driver *driver = build_driver(model);
        if (verbose) printf("Built state.\n");
        if (verbose) timer.stop("Initialization");

        timer.start();
        for (int i = 0 ; i < iterations ; ++i) {
            driver->timestep();
            driver->print_output();
        }

        float time = timer.stop("Total time");
        if (verbose)
            printf("Time averaged over %d iterations: %f\n", iterations, time/iterations);

    } catch (const char* msg) {
        printf("\n\nERROR: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
    return 0;
}

int main(void) {
    Model *model = build_model("izhikevich", true);
    return run_simulation(model, 50, true);
}
