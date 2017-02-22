#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model/model.h"
#include "state/state.h"
#include "util/tools.h"
#include "clock.h"

void print_model(Model *model) {
    printf("Built model.\n");
    printf("  - neurons     : %10d\n", model->get_num_neurons());
    printf("  - layers      : %10d\n", model->get_num_layers());
    printf("  - connections : %10d\n", model->get_num_connections());
    printf("  - weights     : %10d\n", model->get_num_weights());

    for (auto structure : model->get_structures()) {
        for (auto layer : structure->get_layers()) {
            std::cout << layer->structure->name << "->" << layer->name;
            switch (layer->get_type()) {
                case INPUT: std::cout << "\t\tINPUT"; break;
                case INPUT_OUTPUT: std::cout << "\t\tINPUT_OUTPUT"; break;
                case OUTPUT: std::cout << "\t\tOUTPUT"; break;
                case INTERNAL: std::cout << "\t\tINTERNAL"; break;
            }
            std::cout << std::endl;
        }
    }
}

Model* build_self_connected_model(std::string engine_name) {
    Model *model = new Model();
    Structure *structure = model->add_structure("Self-connected", engine_name);

    int rows = 1000;
    int cols = 1000;
    structure->add_layer("a", rows, cols, "random positive", 5);
    structure->connect_layers("a", "a", false, 0, .5, CONVOLUTIONAL, ADD, "7 1");

    structure->add_layer("b", rows, cols, "random positive", 5);
    structure->connect_layers("b", "b", false, 0, .5, CONVOLUTIONAL, ADD, "7 1");

    // Modules
    //structure->add_module("a", "dummy_output", "5");
    //structure->add_module("b", "dummy_output", "5");

    return model;
}

Model* build_arborized_model(std::string engine_name, ConnectionType type) {
    Model *model = new Model();
    Structure *structure = model->add_structure("Arborized", engine_name);

    int rows = 1000;
    int cols = 1000;
    structure->add_layer("a", rows, cols, "random positive", 5);
    structure->connect_layers_expected("a", "b", "random positive" , false, 0, .5, type, ADD, "7 1");

    // Modules
    //structure->add_module("b", "dummy_output", "5");

    return model;
}

Model* build_stress_model(std::string engine_name) {
    Model *model = new Model();
    Structure *structure = model->add_structure("Self-connected", engine_name);

    int size = 800 * 19;
    structure->add_layer("pos", 1, size, "random positive", 5);
    structure->add_layer("neg", 1, size / 4, "random negative", 2);
    structure->connect_layers("pos", "pos", false, 0, .5, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("pos", "neg", false, 0, .5, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("neg", "pos", false, 0, 1, FULLY_CONNECTED, SUB, "");
    structure->connect_layers("neg", "neg", false, 0, 1, FULLY_CONNECTED, SUB, "");

    return model;
}

Model* build_layers_model(std::string engine_name) {
    Model *model = new Model();
    Structure *structure = model->add_structure("Self-connected", engine_name);

    int size = 100;
    structure->add_layer("a", size, size, "random positive", 10);

    structure->add_layer("c", size, size, "random positive");
    structure->connect_layers("a", "c", false, 0, 5, CONVOLUTIONAL, ADD, "5 1");
    structure->add_layer("d", size, size, "random positive");
    structure->connect_layers("a", "d", false, 0, 5, CONVOLUTIONAL, ADD, "5 1");
    structure->add_layer("e", size, size, "random positive");
    structure->connect_layers("a", "e", false, 0, 5, CONVOLUTIONAL, ADD, "5 1");

    structure->add_layer("f", size, size, "random positive");
    structure->connect_layers("c", "f", false, 0, 5, CONVOLUTIONAL, ADD, "5 1");

    structure->add_layer("g", size, size, "random positive");
    structure->connect_layers("d", "g", false, 0, 5, CONVOLUTIONAL, ADD, "5 1");
    structure->connect_layers("f", "g", false, 0, 5, CONVOLUTIONAL, ADD, "5 1");

    structure->add_layer("b", size, size, "random positive", 10);
    structure->connect_layers("f", "b", false, 0, 5, CONVOLUTIONAL, ADD, "5 1");

    // Modules
    structure->add_module("f", "dummy_output", "");
    //structure->add_module("b", "dummy_output", "");

    std::string output_name = "visualizer_output";
    structure->add_module("a", output_name, "");
    structure->add_module("b", output_name, "");
    structure->add_module("c", output_name, "");
    structure->add_module("d", output_name, "");
    structure->add_module("e", output_name, "");
    structure->add_module("f", output_name, "");
    structure->add_module("g", output_name, "");

    return model;
}

Model* build_image_model(std::string engine_name) {
    /* Determine output type */
    //std::string output_name = "print_output";
    std::string output_name = "visualizer_output";

    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("Self-connected", engine_name);

    //const char* image_path = "resources/bird.jpg";
    const char* image_path = "resources/bird-head.jpg";
    //const char* image_path = "resources/pattern.jpg";
    //const char* image_path = "resources/bird-head-small.jpg";
    //const char* image_path = "resources/grid.png";
    structure->add_layer_from_image("photoreceptor", image_path, "default");

    // Vertical line detection
    structure->connect_layers_expected("photoreceptor", "vertical", "default",
        false, 0, 5, CONVOLUTIONAL, ADD,
        "11 1 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 "
        "-5 -5  0  0  5 10  5  0  0 -5 -5 ");
    structure->connect_layers("vertical", "vertical", false, 0, 5, CONVOLUTIONAL, ADD,
        "5 1 "
        "-5  0  5  0 -5 "
        "-5  0  5  0 -5 "
        "-5  0  5  0 -5 "
        "-5  0  5  0 -5 "
        "-5  0  5  0 -5 ");

    // Horizontal line detection
    structure->connect_layers_expected("photoreceptor", "horizontal", "default",
        false, 0, 5, CONVOLUTIONAL, ADD,
        "11 1 "
        "-5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 "
        "-5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 "
        " 0  0  0  0  0  0  0  0  0  0  0 "
        " 0  0  0  0  0  0  0  0  0  0  0 "
        " 5  5  5  5  5  5  5  5  5  5  5 "
        "10 10 10 10 10 10 10 10 10 10 10 "
        " 5  5  5  5  5  5  5  5  5  5  5 "
        " 0  0  0  0  0  0  0  0  0  0  0 "
        " 0  0  0  0  0  0  0  0  0  0  0 "
        "-5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 "
        "-5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 ");
    structure->connect_layers("horizontal", "horizontal", false, 0, 5, CONVOLUTIONAL, ADD,
        "5 1 "
        "-5 -5 -5 -5 -5 "
        " 0  0  0  0  0 "
        " 5  5  5  5  5 "
        " 0  0  0  0  0 "
        "-5 -5 -5 -5 -5 ");

    // Cross detection
    structure->connect_layers_expected("vertical", "cross", "default",
        false, 0, 5, CONVOLUTIONAL, ADD,
        "5 1 "
        "-5  0 10  0 -5 "
        "-5  0 10  0 -5 "
        "-5  0 10  0 -5 "
        "-5  0 10  0 -5 "
        "-5  0 10  0 -5 ");
    structure->connect_layers("horizontal", "cross", false, 0, 5, CONVOLUTIONAL, ADD,
        "5 1 "
        "-5 -5 -5 -5 -5 "
        " 0  0  0  0  0 "
        "10 10 10 10 10 "
        " 0  0  0  0  0 "
        "-5 -5 -5 -5 -5 ");

    // Forward slash
    structure->connect_layers_expected("photoreceptor", "forward_slash", "default",
        false, 0, 5, CONVOLUTIONAL, ADD,
        "9 1 "
        " 0  0  0  0 -5 -5  0  5 10 "
        " 0  0  0 -5 -5  0  5 10  5 "
        " 0  0 -5 -5  0  5 10  5  0 "
        " 0 -5 -5  0  5 10  5  0 -5 "
        "-5 -5  0  5 10  5  0 -5 -5 "
        "-5  0  5 10  5  0 -5 -5  0 "
        " 0  5 10  5  0 -5 -5  0  0 "
        " 5 10  5  0 -5 -5  0  0  0 "
        "10  5  0 -5 -5  0  0  0  0 ");
    structure->connect_layers("forward_slash", "forward_slash", false, 0, 5, CONVOLUTIONAL, ADD,
        "5 1 "
        " 0  0 -5  0  5 "
        " 0 -5  0  5  0 "
        "-5  0  5  0 -5 "
        " 0  5  0 -5  0 "
        " 5  0 -5  0  0 ");

    // Back slash
    structure->connect_layers_expected("photoreceptor", "back_slash", "default",
        false, 0, 5, CONVOLUTIONAL, ADD,
        "9 1 "
        "10  5  0 -5 -5  0  0  0  0 "
        " 5 10  5  0 -5 -5  0  0  0 "
        " 0  5 10  5  0 -5 -5  0  0 "
        "-5  0  5 10  5  0 -5 -5  0 "
        "-5 -5  0  5 10  5  0 -5 -5 "
        " 0 -5 -5  0  5 10  5  0 -5 "
        " 0  0 -5 -5  0  5 10  5  0 "
        " 0  0  0 -5 -5  0  5 10  5 "
        " 0  0  0  0 -5 -5  0  5 10 ");
    structure->connect_layers("back_slash", "back_slash", false, 0, 5, CONVOLUTIONAL, ADD,
        "5 1 "
        " 5  0 -5  0  0 "
        " 0  5  0 -5  0 "
        "-5  0  5  0 -5 "
        " 0 -5  0  5  0 "
        " 0  0 -5  0  5 ");

    // Modules
    structure->add_module("photoreceptor", "image_input", image_path);
    structure->add_module("photoreceptor", output_name, "8");
    structure->add_module("vertical", output_name, "8");
    structure->add_module("horizontal", output_name, "8");
    structure->add_module("forward_slash", output_name, "8");
    structure->add_module("back_slash", output_name, "8");
    //structure->add_module("cross", output_name, "8");

    return model;
}

Model* build_reentrant_image_model(std::string engine_name) {
    /* Determine output type */
    //std::string output_name = "print_output";
    std::string output_name = "visualizer_output";

    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("Self-connected", engine_name);

    //const char* image_path = "resources/bird.jpg";
    const char* image_path = "resources/bird-head.jpg";
    //const char* image_path = "resources/pattern.jpg";
    //const char* image_path = "resources/bird-head-small.jpg";
    //const char* image_path = "resources/grid.png";
    structure->add_layer_from_image("photoreceptor", image_path, "default");

    // Connect first layer to receptor
    structure->connect_layers_matching("photoreceptor", "layer1", "default",
        false, 0, 1, CONVOLUTIONAL, ADD, "21 1 1");

    // Create reentrant pair
    structure->connect_layers_matching("layer1", "layer2", "default",
        false, 0, 1, CONVOLUTIONAL, ADD, "9 1 1", 5);
    structure->connect_layers("layer2", "layer1",
        false, 0, 1, CONVOLUTIONAL, ADD, "9 1 1");

    // Inhibitory self connections
    structure->connect_layers("layer1", "layer1",
        false, 0, 1, CONVOLUTIONAL, SUB, "5 1 10");

    structure->connect_layers("layer2", "layer2",
        false, 0, 1, CONVOLUTIONAL, SUB, "5 1 10");

    // Modules
    structure->add_module("photoreceptor", "image_input", image_path);
    structure->add_module("photoreceptor", output_name, "8");
    structure->add_module("layer1", output_name, "8");
    structure->add_module("layer2", output_name, "8");

    return model;
}

Model* build_alignment_model(std::string engine_name) {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("alignment", engine_name);

    int resolution = 128;
    structure->add_layer("input_layer", 1, 10, "default");
    structure->add_layer("exc_thalamus", resolution, resolution, "low_threshold", 0.5);
    structure->add_layer("inh_thalamus", resolution, resolution, "default");
    structure->add_layer("exc_cortex", resolution, resolution, "thalamo_cortical");
    structure->add_layer("inh_cortex", resolution, resolution, "default");

    structure->connect_layers("input_layer", "exc_thalamus",
        false, 0, 5, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("exc_thalamus", "exc_cortex",
        true, 0, 10, CONVERGENT, ADD, "7 1 0.25");
    structure->connect_layers("exc_cortex", "inh_cortex",
        true, 0, 5, CONVERGENT, ADD, "9 1 0.25");
    structure->connect_layers("exc_cortex", "exc_cortex",
        true, 2, 5, CONVERGENT, ADD, "5 1 0.25");
    structure->connect_layers("inh_cortex", "exc_cortex",
        false, 0, 5, CONVERGENT, DIV, "5 1 5");
    structure->connect_layers("exc_cortex", "inh_thalamus",
        true, 0, 5, CONVERGENT, ADD, "7 1 0.25");
    structure->connect_layers("inh_thalamus", "exc_thalamus",
        false, 0, 5, CONVERGENT, DIV, "5 1 5");

    structure->connect_layers_matching("exc_cortex", "output_layer", "low_threshold",
        true, 40, 0.1, CONVERGENT, ADD, "15 1 0.025");
    structure->connect_layers("output_layer", "exc_cortex",
        false, 40, 1, CONVERGENT, ADD, "15 1 0.5");

    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    structure->add_module("input_layer", "random_input", "10 500");
    structure->add_module("exc_thalamus", output_name, "8");
    structure->add_module("exc_cortex", output_name, "8");
    //structure->add_module("inh_cortex", output_name, "8");
    //structure->add_module("inh_thalamus", output_name, "8");
    structure->add_module("output_layer", output_name, "8");

    return model;
}

Model* build_dendritic_model(std::string engine_name) {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("dendritic", engine_name);

    int resolution = 125;
    structure->add_layer("input_layer1", 1, 10, "default");
    structure->add_layer("input_layer2", 1, 10, "default");
    structure->add_layer("exc_thalamus", resolution, resolution, "low_threshold", 0.5);
    structure->add_layer("inh_thalamus1", resolution, resolution, "default");
    structure->add_layer("inh_thalamus2", resolution, resolution, "default");
    structure->add_layer("exc_cortex", resolution, resolution, "thalamo_cortical");
    structure->add_layer("inh_cortex", resolution, resolution, "default");

    structure->connect_layers("exc_thalamus", "exc_cortex",
        true, 0, 10, CONVERGENT, ADD, "7 1 0.25");
    structure->connect_layers("exc_cortex", "inh_cortex",
        true, 0, 5, CONVERGENT, ADD, "9 1 0.25");
    structure->connect_layers("exc_cortex", "exc_cortex",
        true, 2, 5, CONVERGENT, ADD, "5 1 0.25");
    structure->connect_layers("inh_cortex", "exc_cortex",
        false, 0, 5, CONVERGENT, DIV, "5 1 5");

    // Input branch 1
    auto node1 = structure->spawn_dendritic_node("exc_thalamus");
    structure->connect_layers_internal(node1, "input_layer1",
        false, 0, 5, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("input_layer1", "inh_thalamus1",
        false, 0, 10, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("exc_cortex", "inh_thalamus1",
        true, 0, 5, CONVERGENT, ADD, "7 1 0.25");
    structure->connect_layers_internal(node1, "inh_thalamus1",
        false, 0, 5, CONVERGENT, DIV, "5 1 5");

    // Input branch 2
    auto node2 = structure->spawn_dendritic_node("exc_thalamus");
    structure->connect_layers_internal(node2, "input_layer2",
        false, 0, 5, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("input_layer2", "inh_thalamus2",
        false, 0, 10, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("exc_cortex", "inh_thalamus2",
        true, 0, 5, CONVERGENT, ADD, "7 1 0.25");
    structure->connect_layers_internal(node2, "inh_thalamus2",
        false, 0, 5, CONVERGENT, DIV, "5 1 5");

    structure->connect_layers_matching("exc_cortex", "output_layer", "low_threshold",
        true, 40, 0.1, CONVERGENT, ADD, "15 1 0.025");
    structure->connect_layers("output_layer", "exc_cortex",
        false, 40, 1, CONVERGENT, ADD, "15 1 0.5");

    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    structure->add_module("input_layer1", "random_input", "10 500");
    structure->add_module("input_layer2", "random_input", "10 500");
    structure->add_module("exc_thalamus", output_name, "8");
    structure->add_module("exc_cortex", output_name, "8");
    //structure->add_module("inh_cortex", output_name, "8");
    structure->add_module("inh_thalamus1", output_name, "8");
    structure->add_module("inh_thalamus2", output_name, "8");
    structure->add_module("output_layer", output_name, "8");

    return model;
}

Model* build_hh_model() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("hh", "hodgkin_huxley");

    int resolution = 125;
    structure->add_layer("input_layer", 1, 10, "0");
    structure->add_layer("exc_thalamus", resolution, resolution, "0", 0.5);
    structure->add_layer("inh_thalamus", resolution, resolution, "0");
    structure->add_layer("exc_cortex", resolution, resolution, "0");
    structure->add_layer("inh_cortex", resolution, resolution, "0");

    structure->connect_layers("input_layer", "exc_thalamus",
        false, 0, 5, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("exc_thalamus", "exc_cortex",
        true, 0, 10, CONVERGENT, ADD, "7 1 0.25");
    structure->connect_layers("exc_cortex", "inh_cortex",
        true, 0, 5, CONVERGENT, ADD, "9 1 0.25");
    structure->connect_layers("exc_cortex", "exc_cortex",
        true, 2, 5, CONVERGENT, ADD, "5 1 0.25");
    structure->connect_layers("inh_cortex", "exc_cortex",
        false, 0, 5, CONVERGENT, DIV, "5 1 5");
    structure->connect_layers("exc_cortex", "inh_thalamus",
        true, 0, 5, CONVERGENT, ADD, "7 1 0.25");
    structure->connect_layers("inh_thalamus", "exc_thalamus",
        false, 0, 5, CONVERGENT, DIV, "5 1 5");

    structure->connect_layers_matching("exc_cortex", "output_layer", "0",
        true, 40, 0.1, CONVERGENT, ADD, "15 1 0.025");
        //true, 40, 0.1, CONVERGENT, ADD, "15 1 0.0001");
    structure->connect_layers("output_layer", "exc_cortex",
        false, 40, 1, CONVERGENT, ADD, "15 1 0.5");

    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    structure->add_module("input_layer", "random_input", "10 500");
    structure->add_module("exc_thalamus", output_name, "8");
    structure->add_module("exc_cortex", output_name, "8");
    //structure->add_module("inh_cortex", output_name, "8");
    //structure->add_module("inh_thalamus", output_name, "8");
    structure->add_module("output_layer", output_name, "8");

    return model;
}

Model* build_cc_model(std::string engine_name) {
    /* Construct the model */
    Model *model = new Model();
    std::vector<Structure*> structures;
    int num_structures = 4;

    for (int i = 0 ; i < num_structures ; ++i) {
        Structure *structure = model->add_structure(std::to_string(i), engine_name);

        int resolution = 128;
        structure->add_layer("input_layer", 1, 10, "default");
        structure->add_layer("exc_thalamus", resolution, resolution, "low_threshold", 0.5);
        structure->add_layer("inh_thalamus", resolution, resolution, "default");
        structure->add_layer("exc_cortex", resolution, resolution, "thalamo_cortical");
        structure->add_layer("inh_cortex", resolution, resolution, "default");

        structure->connect_layers("input_layer", "exc_thalamus", false, 0, 5, FULLY_CONNECTED, ADD, "");
        structure->connect_layers("exc_thalamus", "exc_cortex", true, 0, 10, CONVERGENT, ADD, "7 1 0.25");
        structure->connect_layers("exc_cortex", "inh_cortex", true, 0, 5, CONVERGENT, ADD, "9 1 0.25");
        structure->connect_layers("exc_cortex", "exc_cortex", true, 2, 5, CONVERGENT, ADD, "5 1 0.25");
        structure->connect_layers("inh_cortex", "exc_cortex", false, 0, 5, CONVERGENT, DIV, "5 1 10");
        structure->connect_layers("exc_cortex", "inh_thalamus", true, 0, 5, CONVERGENT, ADD, "7 1 0.25");
        structure->connect_layers("inh_thalamus", "exc_thalamus", false, 0, 5, CONVERGENT, DIV, "5 1 10");

        /*
        structure->connect_layers_matching("exc_cortex", "output_layer", "low_threshold",
            true, 20, 0.1, CONVERGENT, ADD, "15 1 0.025");
        structure->connect_layers("output_layer", "exc_cortex",
            true, 20, 0.5, CONVERGENT, ADD, "15 1 0.1");
            //false, 20, 1, ONE_TO_ONE, ADD, "1");
        */

        // Modules
        //std::string output_name = "dummy_output";
        std::string output_name = "visualizer_output";

        structure->add_module("input_layer", "random_input", "10 500");

        structure->add_module("exc_thalamus", output_name, "8");
        structure->add_module("exc_cortex", output_name, "8");
        //structure->add_module("inh_cortex", output_name, "8");
        //structure->add_module("inh_thalamus", output_name, "8");
        //structure->add_module("output_layer", output_name, "8");

        structures.push_back(structure);
    }

    for (int i = 0 ; i < num_structures ; ++i) {
        Structure::connect(
            structures[i],
            "exc_cortex",
            structures[(i+1)%num_structures],
            "exc_cortex",
            //true, 20, 1, ONE_TO_ONE, ADD, "0.1");
            true, 10, 0.1, CONVERGENT, MULT, "9 1 0.01");
            //true, 20, 0.01, FULLY_CONNECTED, ADD, "0.001");
    }

    return model;
}

Model* build_re_model() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("rate_encoding", "rate_encoding");

    int resolution = 128;
    structure->add_layer("in", 1, 8, "");
    structure->add_layer("hid", resolution, resolution, "");
    structure->add_layer("out", resolution, resolution, "");

    structure->connect_layers("in", "hid",
        true, 0, 5, FULLY_CONNECTED, ADD, "");
    structure->connect_layers("hid", "out",
        true, 0, 5, CONVERGENT, ADD, "9 1");

    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    structure->add_module("in", "random_input", "1 10");
    //structure->add_module("hid", output_name, "8");
    structure->add_module("out", output_name, "8");

    return model;
}

void run_simulation(Model *model, int iterations, bool verbose) {
    // Calculate ideal refresh rate, run for iterations
    Clock clock(true);
    clock.run(model, iterations, verbose);

    // Benchmark the network
    // Use max refresh rate possible
    // Run for 100 iterations
    //Clock clock(false);  // No refresh rate synchronization
    //clock.run(model, 100, verbose);
}

void stress_test() {
    Model *model;

    std::cout << "Stress...\n";
    model = build_stress_model("izhikevich");
    print_model(model);
    run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void layers_test() {
    Model *model;

    std::cout << "Layers...\n";
    model = build_layers_model("izhikevich");
    //model = build_layers_model("rate_encoding");
    print_model(model);
    run_simulation(model, 100000, true);
    std::cout << "\n";

    delete model;
}

void reentrant_image_test() {
    Model *model;

    std::cout << "Reentrant Image...\n";
    model = build_reentrant_image_model("izhikevich");
    print_model(model);
    run_simulation(model, 10000, true);
    //run_simulation(model, 100, true);
    //run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void image_test() {
    Model *model;

    std::cout << "Image...\n";
    model = build_image_model("izhikevich");
    //model = build_image_model("rate_encoding");
    print_model(model);
    run_simulation(model, 10000, true);
    //run_simulation(model, 100, true);
    //run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void alignment_test() {
    Model *model;

    std::cout << "Alignment...\n";
    model = build_alignment_model("izhikevich");
    print_model(model);
    run_simulation(model, 1000000, true);
    //run_simulation(model, 100, true);
    //run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void dendritic_test() {
    Model *model;

    std::cout << "Dendritic...\n";
    model = build_dendritic_model("izhikevich");
    print_model(model);
    run_simulation(model, 1000000, true);
    //run_simulation(model, 100, true);
    //run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void hh_test() {
    Model *model;

    std::cout << "Hodgkin-Huxley...\n";
    model = build_hh_model();
    //model = build_alignment_model("hodgkin_huxley");
    print_model(model);
    run_simulation(model, 1000000, true);
    //run_simulation(model, 100, true);
    //run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void cc_test() {
    Model *model;

    std::cout << "Classification couple...\n";
    model = build_cc_model("izhikevich");
    print_model(model);
    run_simulation(model, 1000000, true);
    //run_simulation(model, 100, true);
    //run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void re_test() {
    Model *model;

    std::cout << "Rate encoding......\n";
    model = build_re_model();
    print_model(model);
    run_simulation(model, 1000000, true);
    //run_simulation(model, 100, true);
    //run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void varied_test() {
    Model *model;

    std::cout << "Self connected...\n";
    model = build_self_connected_model("izhikevich");
    print_model(model);
    run_simulation(model, 50, true);
    std::cout << "\n";
    delete model;

    std::cout << "Convergent...\n";
    model = build_arborized_model("izhikevich", CONVERGENT);
    print_model(model);
    run_simulation(model, 50, true);
    std::cout << "\n";
    delete model;

    std::cout << "Convergent convolutional...\n";
    model = build_arborized_model("izhikevich", CONVOLUTIONAL);
    print_model(model);
    run_simulation(model, 50, true);
    std::cout << "\n";
    delete model;
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(NULL));

    try {
        //stress_test();
        //layers_test();
        //image_test();
        //reentrant_image_test();
        alignment_test();
        //dendritic_test();
        //hh_test();
        //cc_test();
        //re_test();
        //varied_test();

        return 0;
    } catch (const char* msg) {
        printf("\n\nERROR: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
