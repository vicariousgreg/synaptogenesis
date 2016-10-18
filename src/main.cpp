#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model/model.h"
#include "state/state.h"
#include "tools.h"
#include "clock.h"

static Timer timer = Timer();

void print_model(Model *model) {
    printf("Built model.\n");
    printf("  - neurons     : %10d\n", model->num_neurons);
    printf("  - layers      : %10d\n", model->layers.size());
    printf("  - connections : %10d\n", model->connections.size());
    int num_weights = 0;
    for (int i = 0; i < model->connections.size() ; ++i)
        if (model->connections[i]->parent == NULL)
            num_weights += model->connections[i]->num_weights;
    printf("  - weights     : %10d\n", num_weights);
}

Model* build_arborized_model(std::string driver_name, bool verbose, ConnectionType type) {
    Model *model = new Model(driver_name);

    int rows = 1000;
    int cols = 1000;
    int a = model->add_layer(rows, cols, "random positive");
    model->connect_layers_expected(a, "random positive" , true, 0, .5, type, ADD, "7 1");
    model->add_input(a, "random", "5");

    if (verbose) print_model(model);
    model->rearrange();
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

    if (verbose) print_model(model);
    model->rearrange();
    return model;
}

Model* build_image_model(std::string driver_name, bool verbose) {
    /* Determine output type */
    std::string output_name;
    if (driver_name == "izhikevich")
        output_name = "print_spike";
    else if (driver_name == "rate_encoding")
        output_name = "print_float";

    /* Construct the model */
    Model *model = new Model(driver_name);

    //const char* image_path = "resources/bird-head.jpg";
    const char* image_path = "resources/bird-head-small.jpg";
    int receptor = model->add_layer_from_image(image_path, "default");
    model->add_input(receptor, "image", image_path);
    model->add_output(receptor, output_name, "24");

    // Vertical line detection
    int vertical = model->connect_layers_expected(receptor, "default",
        true, 0, 5, CONVERGENT_CONVOLUTIONAL, ADD,
        "5 1 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5");
    //model->add_output(vertical, output_name, "31");

    // Horizontal line detection
    int horizontal = model->connect_layers_expected(receptor, "default",
        true, 0, 5, CONVERGENT_CONVOLUTIONAL, ADD,
        "5 1 "
        "-5 -5 -5 -5 -5 "
        "0 0 0 0 0 "
        "10 10 10 10 10 "
        "0 0 0 0 0 "
        "-5 -5 -5 -5 -5");
    //model->add_output(horizontal, output_name, "31");

    // Cross detection
    int cross = model->connect_layers_expected(vertical, "default",
        true, 0, 5, CONVERGENT_CONVOLUTIONAL, ADD,
        "5 1 "
        "-.5  -.5 1  -.5 -.5 "
        "-.5  5   10 5   -.5 "
        "-1   -.5 15 -.5 -1 "
        "-.5  5   10 5   -.5 "
        "-.5  -.5 1  -.5 -.5");
    model->connect_layers(horizontal, cross, true, 0, 5, CONVERGENT_CONVOLUTIONAL, ADD,
        "5 1 "
        "-.5  -.5  -1   -.5  -.5 "
        "-.5    5   -.5 5   -.5 "
        "1      10  15  10  1 "
        "-.5    5   -.5 5   -.5 "
        "-.5  -.5  -1  -.5  -.5");
    //model->add_output(cross, output_name, "16");

    if (verbose) print_model(model);
    model->rearrange();
    return model;
}

void run_simulation(Model *model, int iterations, bool verbose) {
    // Seed random number generator
    srand(time(NULL));

    Clock clock(10);

    clock.run(model, iterations, verbose);
#ifdef PARALLEL
    check_memory();
#endif
}

void stress_test() {
    Model *model;

    std::cout << "Stress...\n";
    model = build_stress_model("izhikevich", true);
    run_simulation(model, 50, true);
    std::cout << "\n";
}

void image_test() {
    Model *model;

    std::cout << "Image...\n";
    model = build_image_model("izhikevich", true);
    //model = build_image_model("rate_encoding", true);
    run_simulation(model, 500, true);
    std::cout << "\n";
}

void varied_test() {
    Model *model;

    std::cout << "Convergent...\n";
    model = build_arborized_model("izhikevich", true, CONVERGENT);
    run_simulation(model, 50, true);
    std::cout << "\n";

    std::cout << "Convergent convolutional...\n";
    model = build_arborized_model("izhikevich", true, CONVERGENT_CONVOLUTIONAL);
    run_simulation(model, 50, true);
    std::cout << "\n";

    std::cout << "Divergent...\n";
    model = build_arborized_model("izhikevich", true, DIVERGENT);
    run_simulation(model, 50, true);
    std::cout << "\n";

    std::cout << "Divergent convolutional...\n";
    model = build_arborized_model("izhikevich", true, DIVERGENT_CONVOLUTIONAL);
    run_simulation(model, 50, true);
    std::cout << "\n";
}

int main(void) {
    try {
        //stress_test();
        image_test();
        //varied_test();

        return 0;
    } catch (const char* msg) {
        printf("\n\nERROR: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
