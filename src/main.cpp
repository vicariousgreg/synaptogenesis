#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model/model.h"
#include "state/state.h"
#include "util/tools.h"
#include "clock.h"

static Timer timer = Timer();

void print_model(Model *model) {
    printf("Built model.\n");
    printf("  - neurons     : %10d\n", model->num_neurons);
    printf("  - layers      : %10d\n", model->all_layers.size());
    printf("  - connections : %10d\n", model->connections.size());
    int num_weights = 0;
    for (int i = 0; i < model->connections.size() ; ++i)
        if (model->connections[i]->parent == NULL)
            num_weights += model->connections[i]->num_weights;
    printf("  - weights     : %10d\n", num_weights);
}

Model* build_self_connected_model(std::string engine_name, bool verbose) {
    Model *model = new Model(engine_name);
    Structure *structure = new Structure("Self-connected");

    int rows = 1000;
    int cols = 1000;
    structure->add_layer("a", rows, cols, "random positive");
    structure->connect_layers("a", "a", true, 0, .5, CONVERGENT_CONVOLUTIONAL, ADD, "7 1");
    structure->add_module("a", "random_input", "5");
    //structure->add_module("a", "dummy_output", "5");

    structure->add_layer("b", rows, cols, "random positive");
    structure->connect_layers("b", "b", true, 0, .5, DIVERGENT_CONVOLUTIONAL, ADD, "7 1");
    structure->add_module("b", "random_input", "5");
    //structure->add_module("b", "dummy_output", "5");

    model->add_structure(structure);
    if (verbose) print_model(model);
    return model;
}

Model* build_arborized_model(std::string engine_name, bool verbose, ConnectionType type) {
    Model *model = new Model(engine_name);
    Structure *structure = new Structure("Arborized");

    int rows = 1000;
    int cols = 1000;
    structure->add_layer("a", rows, cols, "random positive");
    structure->connect_layers_expected("a", "b", "random positive" , true, 0, .5, type, ADD, "7 1");
    structure->add_module("a", "random_input", "5");
    //structure->add_module("b", "dummy_output", "5");

    model->add_structure(structure);
    if (verbose) print_model(model);
    return model;
}

Model* build_stress_model(std::string engine_name, bool verbose) {
    Model *model = new Model(engine_name);
    Structure *structure = new Structure("Self-connected");

    int size = 800 * 19;
    structure->add_layer("pos", 1, size, "random positive");
    structure->add_layer("neg", 1, size / 4, "random negative");
    structure->connect_layers("pos", "pos", true, 0, .5, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("pos", "neg", true, 0, .5, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("neg", "pos", true, 0, 1, FULLY_CONNECTED, SUB, "");
    structure->connect_layers("neg", "neg", true, 0, 1, FULLY_CONNECTED, SUB, "");
    structure->add_module("pos", "random_input", "5");
    structure->add_module("neg", "random_input", "2");

    model->add_structure(structure);
    if (verbose) print_model(model);
    return model;
}

Model* build_layers_model(std::string engine_name, bool verbose) {
    Model *model = new Model(engine_name);
    Structure *structure = new Structure("Self-connected");

    int size = 1000;
    structure->add_layer("a", 1, size, "random positive");
    structure->add_module("a", "dummy_input", "");

    structure->add_layer("c", 10, size, "random positive");
    structure->connect_layers("a", "c", true, 0, .5, FULLY_CONNECTED, ADD, "");
    structure->add_layer("d", 10, size, "random positive");
    structure->connect_layers("a", "d", true, 0, .5, FULLY_CONNECTED, ADD, "");
    structure->add_layer("e", 10, size, "random positive");
    structure->connect_layers("a", "e", true, 0, .5, FULLY_CONNECTED, ADD, "");

    structure->add_layer("f", 1, size, "random positive");
    structure->connect_layers("c", "f", true, 0, .5, FULLY_CONNECTED, ADD, "");
    structure->add_module("f", "dummy_output", "");

    structure->add_layer("g", 1, size, "random positive");
    structure->connect_layers("d", "g", true, 0, .5, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("f", "g", true, 0, .5, FULLY_CONNECTED, ADD, "");

    structure->add_layer("b", 1, size, "random positive");
    structure->connect_layers("f", "b", true, 0, .5, FULLY_CONNECTED, ADD, "");
    structure->add_module("b", "dummy_input", "");
    structure->add_module("b", "dummy_output", "");

    model->add_structure(structure);
    if (verbose) print_model(model);
    return model;
}

Model* build_image_model(std::string engine_name, bool verbose) {
    /* Determine output type */
    std::string output_name;
    //output_name = "print_output";
    output_name = "visualizer_output";

    /* Construct the model */
    Model *model = new Model(engine_name);
    Structure *structure = new Structure("Self-connected");

    //const char* image_path = "resources/bird.jpg";
    const char* image_path = "resources/bird-head.jpg";
    //const char* image_path = "resources/bird-head-small.jpg";
    //const char* image_path = "resources/grid.png";
    structure->add_layer_from_image("photoreceptor", image_path, "default");
    structure->add_module("photoreceptor", "image_input", image_path);
    structure->add_module("photoreceptor", output_name, "24");

    // Vertical line detection
    structure->connect_layers_expected("photoreceptor", "vertical", "default",
        true, 0, 5, CONVERGENT_CONVOLUTIONAL, ADD,
        "5 1 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5 "
        "-5 0 10 0 -5");
    //structure->add_module("vertical", output_name, "24");

    // Horizontal line detection
    structure->connect_layers_expected("photoreceptor", "horizontal", "default",
        true, 0, 5, CONVERGENT_CONVOLUTIONAL, ADD,
        "5 1 "
        "-5 -5 -5 -5 -5 "
        "0 0 0 0 0 "
        "10 10 10 10 10 "
        "0 0 0 0 0 "
        "-5 -5 -5 -5 -5");
    //structure->add_module("horizontal", output_name, "24");

    // Cross detection
    structure->connect_layers_expected("vertical", "cross", "default",
        true, 0, 5, CONVERGENT_CONVOLUTIONAL, ADD,
        "5 1 "
        "-.5  -.5 1  -.5 -.5 "
        "-.5  5   10 5   -.5 "
        "-1   -.5 15 -.5 -1 "
        "-.5  5   10 5   -.5 "
        "-.5  -.5 1  -.5 -.5");
    structure->connect_layers("horizontal", "cross", true, 0, 5, CONVERGENT_CONVOLUTIONAL, ADD,
        "5 1 "
        "-.5  -.5  -1   -.5  -.5 "
        "-.5    5   -.5 5   -.5 "
        "1      10  15  10  1 "
        "-.5    5   -.5 5   -.5 "
        "-.5  -.5  -1  -.5  -.5");
    //structure->add_module("cross", output_name, "24");

    model->add_structure(structure);
    if (verbose) print_model(model);
    return model;
}

void run_simulation(Model *model, int iterations, bool verbose) {
    Clock clock(10);
    //Clock clock;  // No refresh rate synchronization
    //clock.run(model, iterations, 8, verbose);
    clock.run(model, iterations, 1, verbose);
}

void stress_test() {
    Model *model;

    std::cout << "Stress...\n";
    model = build_stress_model("izhikevich", true);
    run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void layers_test() {
    Model *model;

    std::cout << "Layers...\n";
    model = build_layers_model("izhikevich", true);
    //model = build_layers_model("rate_encoding", true);
    run_simulation(model, 500, true);
    std::cout << "\n";

    delete model;
}

void image_test() {
    Model *model;

    std::cout << "Image...\n";
    model = build_image_model("izhikevich", true);
    //model = build_image_model("rate_encoding", true);
    run_simulation(model, 10000, true);
    //run_simulation(model, 100, true);
    //run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void varied_test() {
    Model *model;

    std::cout << "Self connected...\n";
    model = build_self_connected_model("izhikevich", true);
    run_simulation(model, 50, true);
    std::cout << "\n";
    delete model;

    std::cout << "Convergent...\n";
    model = build_arborized_model("izhikevich", true, CONVERGENT);
    run_simulation(model, 50, true);
    std::cout << "\n";
    delete model;

    std::cout << "Convergent convolutional...\n";
    model = build_arborized_model("izhikevich", true, CONVERGENT_CONVOLUTIONAL);
    run_simulation(model, 50, true);
    std::cout << "\n";
    delete model;

    std::cout << "Divergent...\n";
    model = build_arborized_model("izhikevich", true, DIVERGENT);
    run_simulation(model, 50, true);
    std::cout << "\n";
    delete model;

    std::cout << "Divergent convolutional...\n";
    model = build_arborized_model("izhikevich", true, DIVERGENT_CONVOLUTIONAL);
    run_simulation(model, 50, true);
    std::cout << "\n";
    delete model;
}

int main(void) {
    // Seed random number generator
    srand(time(NULL));

    try {
        //stress_test();
        //layers_test();
        image_test();
        //varied_test();

        return 0;
    } catch (const char* msg) {
        printf("\n\nERROR: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
