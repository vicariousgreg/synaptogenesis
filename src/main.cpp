#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model/model.h"
#include "state/state.h"
#include "driver/driver.h"
#include "tools.h"

static Timer timer = Timer();

Model* build_arborized_model(std::string driver_name, bool verbose, ConnectionType type) {
    Model *model = new Model(driver_name);

    int diff = 6;
    switch (type) {
        case (CONVERGENT):
            diff = -diff;
            break;
        case (CONVERGENT_CONVOLUTIONAL):
            diff = -diff;
            break;
        case (DIVERGENT):
            break;
        case (DIVERGENT_CONVOLUTIONAL):
            break;
    }

    int rows = 1000;
    int cols = 1000;
    int a = model->add_layer(rows, cols, "random positive");
    int b = model->add_layer(rows+diff, cols+diff, "random negative");
    model->connect_layers(a, b, true, 0, .5, type, ADD, "7 1");
    model->add_input(a, "random", "5");

    if (verbose) {
        printf("Built model.\n");
        printf("  - neurons     : %10d\n", model->num_neurons);
        printf("  - layers      : %10d\n", model->num_layers);
        printf("  - connections : %10d\n", model->num_connections);
        int num_weights = 0;
        for (int i = 0; i < model->num_connections ; ++i)
            if (model->connections[i]->parent == -1)
                num_weights += model->connections[i]->num_weights;
        printf("  - weights     : %10d\n", num_weights);
    }

    return model;
}

Model* build_stress_model(std::string driver_name, bool verbose) {
    Model *model = new Model(driver_name);

    int size = 800 * 20;
    int pos = model->add_layer(1, size, "random positive");
    int neg = model->add_layer(1, size / 4, "random negative");
    model->connect_layers(pos, pos, true, 0, .5, FULLY_CONNECTED, ADD, "");
    model->connect_layers(pos, neg, true, 0, .5, FULLY_CONNECTED, ADD, "");
    model->connect_layers(neg, pos, true, 0, 1, FULLY_CONNECTED, SUB, "");
    model->connect_layers(neg, neg, true, 0, 1, FULLY_CONNECTED, SUB, "");
    model->add_input(pos, "random", "5");
    model->add_input(neg, "random", "2");

    if (verbose) {
        printf("Built model.\n");
        printf("  - neurons     : %10d\n", model->num_neurons);
        printf("  - layers      : %10d\n", model->num_layers);
        printf("  - connections : %10d\n", model->num_connections);
        int num_weights = 0;
        for (int i = 0; i < model->num_connections ; ++i)
            if (model->connections[i]->parent == -1)
                num_weights += model->connections[i]->num_weights;
        printf("  - weights     : %10d\n", num_weights);
    }

    return model;
}

Model* build_image_model(std::string driver_name, bool verbose) {
    /* Construct the model */
    Model *model = new Model(driver_name);

    //int image_size = 373;
    int image_size = 50;

    int receptor = model->add_layer(image_size, image_size, "default");
    //model->add_input(receptor, "image", "resources/bird-head-small.jpg");
    model->add_input(receptor, "image", "resources/grid.png");
    //model->add_output(receptor, "print_spike", "");

    // Vertical line detection
    //int vertical = model->add_layer(image_size+4, image_size+4, "default");
    int vertical = model->add_layer(image_size-4, image_size-4, "default");
    model->connect_layers(receptor, vertical, true, 0, 5, CONVERGENT_CONVOLUTIONAL, ADD,
        "5 1 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5");
    //model->add_output(vertical, "print_spike", "");

    // Horizontal line detection
    int horizontal = model->add_layer(image_size-4, image_size-4, "default");
    model->connect_layers(receptor, horizontal, true, 0, 5, CONVERGENT_CONVOLUTIONAL, ADD,
        "5 1 "
        "-5 -5 -5 -5 -5 "
        "0 0 0 0 0 "
        "10 10 10 10 10 "
        "0 0 0 0 0 "
        "-5 -5 -5 -5 -5");
    //model->add_output(horizontal, "print_spike", "");

    // Cross detection
    int cross = model->add_layer(image_size-4, image_size-4, "default");
    //int cross = model->add_layer(image_size-4+2, image_size-4+2, "default");
    //model->connect_layers(vertical, cross, true, 0, 5, DIVERGENT, ADD, "3 1");
    model->connect_layers(vertical, cross, true, 0, 5, ONE_TO_ONE, ADD, "10");
    model->connect_layers(horizontal, cross, true, 0, 5, ONE_TO_ONE, ADD, "10");
    model->add_output(cross, "print_spike", "");

    if (verbose) {
        printf("Built model.\n");
        printf("  - neurons     : %10d\n", model->num_neurons);
        printf("  - layers      : %10d\n", model->num_layers);
        printf("  - connections : %10d\n", model->num_connections);
        int num_weights = 0;
        for (int i = 0; i < model->num_connections ; ++i)
            if (model->connections[i]->parent == -1)
                num_weights += model->connections[i]->num_weights;
        printf("  - weights     : %10d\n", num_weights);
    }
    return model;
}

void run_simulation(Model *model, int iterations, bool verbose) {
    // Seed random number generator
    srand(time(NULL));

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
#ifdef PARALLEL
    check_memory();
#endif
}

int main(void) {
    try {
        Model *model;
        /*
        std::cout << "Stress...\n";
        model = build_stress_model("izhikevich", true);
        run_simulation(model, 50, true);
        std::cout << "\n";
        */

        /*
        std::cout << "Image...\n";
        model = build_image_model("izhikevich", true);
        run_simulation(model, 50, true);
        std::cout << "\n";
        */

        ///*
        std::cout << "Convergent...\n";
        model = build_arborized_model("izhikevich", true, CONVERGENT);
        run_simulation(model, 50, true);
        std::cout << "\n";
        //*/

        ///*
        std::cout << "Convergent convolutional...\n";
        model = build_arborized_model("izhikevich", true, CONVERGENT_CONVOLUTIONAL);
        run_simulation(model, 50, true);
        std::cout << "\n";
        //*/

        ///*
        std::cout << "Divergent...\n";
        model = build_arborized_model("izhikevich", true, DIVERGENT);
        run_simulation(model, 50, true);
        std::cout << "\n";
        //*/

        ///*
        std::cout << "Divergent convolutional...\n";
        model = build_arborized_model("izhikevich", true, DIVERGENT_CONVOLUTIONAL);
        run_simulation(model, 50, true);
        std::cout << "\n";
        //*/

        return 0;
    } catch (const char* msg) {
        printf("\n\nERROR: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
