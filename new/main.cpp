#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "model.h"
#include "driver.h"
#include "izhikevich.h"

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

Driver* build_driver(Model* model) {
    /* Construct the driver */
    Driver *driver = new Izhikevich();
    driver->build(model);
    return driver;
}

int main(void) {
    // Seed random number generator
    srand(time(NULL));

    try {
        Model *model = build_model();
        printf("Built model.\n");
        printf("  - neurons     : %10d\n", model->num_neurons);
        printf("  - layers      : %10d\n", model->num_layers);
        printf("  - connections : %10d\n", model->num_connections);

        Driver *driver = build_driver(model);
        printf("Built driver.\n");

        //int iterations = 500;
    } catch (const char* msg) {
        printf("\n\nERROR: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }

    return 0;
}
