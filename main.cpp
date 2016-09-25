#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "model.h"
#include "driver.h"
#include "izhikevich_driver.h"
#include "tools.h"

Model* build_model() {
    /* Construct the model */
    Model *model = new Model();
    int size = 800 * 20;

    int pos = model->add_layer(size, "random positive");
    int neg = model->add_layer(size / 4, "random negative");
    model->connect_layers(pos, pos, true, 0, .5, FULLY_CONNECTED, ADD);
    model->connect_layers(pos, neg, true, 0, .5, FULLY_CONNECTED, ADD);
    model->connect_layers(neg, pos, true, 0, 1, FULLY_CONNECTED, SUB);
    model->connect_layers(neg, neg, true, 0, 1, FULLY_CONNECTED, SUB);

    return model;
}

IzhikevichDriver* build_driver(Model* model) {
    /* Construct the driver */
    IzhikevichDriver *driver = new IzhikevichDriver();
    driver->build(model);
    return driver;
}

/* Prints a line for a timestep containing markers for neuron spikes.
 * If a neuron spikes, an asterisk will be printed.  Otherwise, a space */
void print_spikes(IzhikevichDriver *driver) {
    int* spikes = driver->get_spikes();
    for (int nid = 0 ; nid < driver->model->num_neurons ; ++nid) {
        char c = (spikes[nid] % 2) ? '*' : ' ';
        std::cout << c;
    }
    std::cout << "|\n";
}

/* Prints a line for a timestep containing neuron currents */
void print_currents(IzhikevichDriver *driver) {
    float* currents = driver->get_current();
    for (int nid = 0 ; nid < driver->model->num_neurons ; ++nid) {
        std::cout << currents[nid] << " " ;
    }
    std::cout << "|\n";
}

int main(void) {
    // Seed random number generator
    srand(time(NULL));
    Timer timer = Timer();

    try {
        // Start timer
        timer.start();

        Model *model = build_model();
        printf("Built model.\n");
        printf("  - neurons     : %10d\n", model->num_neurons);
        printf("  - layers      : %10d\n", model->num_layers);
        printf("  - connections : %10d\n", model->num_connections);

        IzhikevichDriver *driver = build_driver(model);
        printf("Built driver.\n");
        timer.stop("Initialization");

        timer.start();
        int iterations = 50;
        for (int i = 0 ; i < iterations ; ++i) {
            driver->randomize_current(0, 5);
            driver->randomize_current(1, 2);
            driver->timestep();
            //print_currents(driver);
            //print_spikes(driver);
        }

        float time = timer.stop("Total time");
        printf("Time averaged over %d iterations: %f\n", iterations, time/iterations);
    } catch (const char* msg) {
        printf("\n\nERROR: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }

    return 0;
}
