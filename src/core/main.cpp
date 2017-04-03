#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model/model.h"
#include "model/weight_config.h"
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
            if (layer->is_input()) std::cout << "\t\tINPUT";
            if (layer->is_output()) std::cout << "\t\tOUTPUT";
            if (layer->is_expected()) std::cout << "\t\tOUTPUT";
            std::cout << std::endl;
        }
    }
}

Model* build_stress_model(NeuralModel neural_model) {
    Model *model = new Model();
    Structure *structure = model->add_structure("Self-connected");

    int size = 800 * 19;
    structure->add_layer(LayerConfig("pos",
        neural_model, 1, size, "random positive", 5));
    structure->add_layer(LayerConfig("neg",
        neural_model, 1, size / 4, "random negative", 2));
    structure->connect_layers("pos", "pos",
        ConnectionConfig(false, 0, .5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(0.5)));
    structure->connect_layers("pos", "neg",
        ConnectionConfig(false, 0, .5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(0.5)));
    structure->connect_layers("neg", "pos",
        ConnectionConfig(false, 0, 1, FULLY_CONNECTED, SUB,
            new RandomWeightConfig(1)));
    structure->connect_layers("neg", "neg",
        ConnectionConfig(false, 0, 1, FULLY_CONNECTED, SUB,
            new RandomWeightConfig(1)));

    return model;
}

Model* build_image_model(NeuralModel neural_model) {
    /* Determine output type */
    //std::string output_name = "print_output";
    std::string output_name = "visualizer_output";

    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("Self-connected");

    //const char* image_path = "resources/bird.jpg";
    const char* image_path = "resources/bird-head.jpg";
    //const char* image_path = "resources/pattern.jpg";
    //const char* image_path = "resources/bird-head-small.jpg";
    //const char* image_path = "resources/grid.png";
    structure->add_layer_from_image(image_path,
        LayerConfig("photoreceptor", neural_model, "default"));

    // Vertical line detection
    structure->connect_layers_expected("photoreceptor",
        LayerConfig("vertical", neural_model, "default"),
        ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
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
                "-5 -5  0  0  5 10  5  0  0 -5 -5 "),
            ArborizedConfig(11,1)));
    structure->connect_layers("vertical", "vertical",
        ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                "-5  0  5  0 -5 "
                "-5  0  5  0 -5 "
                "-5  0  5  0 -5 "
                "-5  0  5  0 -5 "
                "-5  0  5  0 -5 "),
            ArborizedConfig(5,1)));

    // Horizontal line detection
    structure->connect_layers_expected("photoreceptor",
        LayerConfig("horizontal", neural_model, "default"),
        ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
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
                "-5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 "),
            ArborizedConfig(11,1)));
    structure->connect_layers("horizontal", "horizontal",
        ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                "-5 -5 -5 -5 -5 "
                " 0  0  0  0  0 "
                " 5  5  5  5  5 "
                " 0  0  0  0  0 "
                "-5 -5 -5 -5 -5 "),
            ArborizedConfig(5,1)));

    // Cross detection
    structure->connect_layers_expected("vertical",
        LayerConfig("cross", neural_model, "default"),
        ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                "-5  0 10  0 -5 "
                "-5  0 10  0 -5 "
                "-5  0 10  0 -5 "
                "-5  0 10  0 -5 "
                "-5  0 10  0 -5 "),
            ArborizedConfig(5,1)));
    structure->connect_layers("horizontal", "cross",
        ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                "-5 -5 -5 -5 -5 "
                " 0  0  0  0  0 "
                "10 10 10 10 10 "
                " 0  0  0  0  0 "
                "-5 -5 -5 -5 -5 "),
            ArborizedConfig(5,1)));

    // Forward slash
    structure->connect_layers_expected("photoreceptor",
        LayerConfig("forward_slash", neural_model, "default"),
        ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                " 0  0  0  0 -5 -5  0  5 10 "
                " 0  0  0 -5 -5  0  5 10  5 "
                " 0  0 -5 -5  0  5 10  5  0 "
                " 0 -5 -5  0  5 10  5  0 -5 "
                "-5 -5  0  5 10  5  0 -5 -5 "
                "-5  0  5 10  5  0 -5 -5  0 "
                " 0  5 10  5  0 -5 -5  0  0 "
                " 5 10  5  0 -5 -5  0  0  0 "
                "10  5  0 -5 -5  0  0  0  0 "),
            ArborizedConfig(9,1)));
    structure->connect_layers("forward_slash", "forward_slash",
        ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                " 0  0 -5  0  5 "
                " 0 -5  0  5  0 "
                "-5  0  5  0 -5 "
                " 0  5  0 -5  0 "
                " 5  0 -5  0  0 "),
            ArborizedConfig(5,1)));

    // Back slash
    structure->connect_layers_expected("photoreceptor",
        LayerConfig("back_slash", neural_model, "default"),
            ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                "10  5  0 -5 -5  0  0  0  0 "
                " 5 10  5  0 -5 -5  0  0  0 "
                " 0  5 10  5  0 -5 -5  0  0 "
                "-5  0  5 10  5  0 -5 -5  0 "
                "-5 -5  0  5 10  5  0 -5 -5 "
                " 0 -5 -5  0  5 10  5  0 -5 "
                " 0  0 -5 -5  0  5 10  5  0 "
                " 0  0  0 -5 -5  0  5 10  5 "
                " 0  0  0  0 -5 -5  0  5 10 "),
            ArborizedConfig(9,1)));
    structure->connect_layers("back_slash", "back_slash",
        ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                " 5  0 -5  0  0 "
                " 0  5  0 -5  0 "
                "-5  0  5  0 -5 "
                " 0 -5  0  5  0 "
                " 0  0 -5  0  5 "),
            ArborizedConfig(5,1)));

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

Model* build_reentrant_image_model(NeuralModel neural_model) {
    /* Determine output type */
    //std::string output_name = "print_output";
    std::string output_name = "visualizer_output";

    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("Self-connected");

    //const char* image_path = "resources/bird.jpg";
    const char* image_path = "resources/bird-head.jpg";
    //const char* image_path = "resources/pattern.jpg";
    //const char* image_path = "resources/bird-head-small.jpg";
    //const char* image_path = "resources/grid.png";
    structure->add_layer_from_image(image_path,
        LayerConfig("photoreceptor", neural_model, "default"));

    // Connect first layer to receptor
    structure->connect_layers_matching("photoreceptor",
        LayerConfig("layer1", neural_model, "default"),
        ConnectionConfig(false, 0, 1, CONVOLUTIONAL, ADD,
            new RandomWeightConfig(1),
            ArborizedConfig(21,1)));

    // Create reentrant pair
    structure->connect_layers_matching("layer1",
        LayerConfig("layer2", neural_model, "default", 5),
        ConnectionConfig(false, 0, 1, CONVOLUTIONAL, ADD,
            new RandomWeightConfig(1),
            ArborizedConfig(9,1)));
    structure->connect_layers("layer2","layer1",
        ConnectionConfig(false, 0, 1, CONVOLUTIONAL, ADD,
            new RandomWeightConfig(1),
            ArborizedConfig(9,1)));

    // Inhibitory self connections
    structure->connect_layers("layer1", "layer1",
        ConnectionConfig(false, 0, 1, CONVOLUTIONAL, SUB,
            new RandomWeightConfig(10),
            ArborizedConfig(5,1)));

    structure->connect_layers("layer2", "layer2",
        ConnectionConfig(false, 0, 1, CONVOLUTIONAL, SUB,
            new RandomWeightConfig(10),
            ArborizedConfig(5,1)));

    // Modules
    structure->add_module("photoreceptor", "image_input", image_path);
    structure->add_module("photoreceptor", output_name, "8");
    structure->add_module("layer1", output_name, "8");
    structure->add_module("layer2", output_name, "8");

    return model;
}

Model* build_alignment_model(NeuralModel neural_model) {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("alignment");

    int resolution = 256;
    structure->add_layer(LayerConfig("input_layer",
        neural_model, 1, 10, "default"));
    structure->add_layer(LayerConfig("exc_thalamus",
        neural_model, resolution, resolution, "low_threshold", 0.5));
    structure->add_layer(LayerConfig("inh_thalamus",
        neural_model, resolution, resolution, "default"));
    structure->add_layer(LayerConfig("exc_cortex",
        neural_model, resolution, resolution, "thalamo_cortical"));
    structure->add_layer(LayerConfig("inh_cortex",
        neural_model, resolution, resolution, "default"));

    structure->connect_layers("input_layer", "exc_thalamus",
        ConnectionConfig(true, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(5)));
        //ConnectionConfig(true, 0, 5, DIVERGENT, ADD,
        //    ArborizedConfig(36,22)));
    structure->connect_layers("exc_thalamus", "exc_cortex",
        ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(7,1)));
    structure->connect_layers("exc_cortex", "inh_cortex",
        ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(9,1)));
    structure->connect_layers("exc_cortex", "exc_cortex",
        ConnectionConfig(true, 2, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(5,1)));
    structure->connect_layers("inh_cortex", "exc_cortex",
        ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            ArborizedConfig(5,1)));
    structure->connect_layers("exc_cortex", "inh_thalamus",
        ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(7,1)));
    structure->connect_layers("inh_thalamus", "exc_thalamus",
        ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            ArborizedConfig(5,1)));

    structure->connect_layers_matching("exc_cortex",
        LayerConfig("output_layer", neural_model, "low_threshold"),
        ConnectionConfig(true, 40, 0.1, CONVERGENT, ADD,
            new RandomWeightConfig(0.025),
            ArborizedConfig(15,1)));
    structure->connect_layers("output_layer", "exc_cortex",
        ConnectionConfig(false, 40, 1, CONVERGENT, ADD,
            new RandomWeightConfig(0.5),
            ArborizedConfig(15,1)));

    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    structure->add_module("input_layer", "random_input", "10 500");
    structure->add_module("exc_cortex", output_name, "8");
    structure->add_module("exc_thalamus", output_name, "8");
    //structure->add_module("inh_cortex", output_name, "8");
    //structure->add_module("inh_thalamus", output_name, "8");
    structure->add_module("output_layer", output_name, "8");
    /*
    */

    return model;
}

Model* build_dendritic_model(NeuralModel neural_model) {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("dendritic");

    int resolution = 128;
    structure->add_layer(LayerConfig("input_layer1",
        neural_model, 1, 10, "default"));
    structure->add_layer(LayerConfig("input_layer2",
        neural_model, 1, 10, "default"));
    structure->add_layer(LayerConfig("exc_thalamus",
        neural_model, resolution, resolution, "low_threshold", 0.5));
    structure->add_layer(LayerConfig("inh_thalamus1",
        neural_model, resolution, resolution, "default"));
    structure->add_layer(LayerConfig("inh_thalamus2",
        neural_model, resolution, resolution, "default"));
    structure->add_layer(LayerConfig("exc_cortex",
        neural_model, resolution, resolution, "thalamo_cortical"));
    structure->add_layer(LayerConfig("inh_cortex",
        neural_model, resolution, resolution, "default"));

    structure->connect_layers("exc_thalamus", "exc_cortex",
        ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(7,1)));
    structure->connect_layers("exc_cortex", "inh_cortex",
        ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(9,1)));
    structure->connect_layers("exc_cortex", "exc_cortex",
        ConnectionConfig(true, 2, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(5,1)));
    structure->connect_layers("inh_cortex", "exc_cortex",
        ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            ArborizedConfig(5,1)));

    // Input branch 1
    auto node1 = structure->spawn_dendritic_node("exc_thalamus");
    structure->connect_layers_internal(node1, "input_layer1",
        ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(5)));
    structure->connect_layers("input_layer1", "inh_thalamus1",
        ConnectionConfig(false, 0, 10, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(10)));
    structure->connect_layers("exc_cortex", "inh_thalamus1",
        ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(7,1)));
    structure->connect_layers_internal(node1, "inh_thalamus1",
        ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            ArborizedConfig(5,1)));

    // Input branch 2
    auto node2 = structure->spawn_dendritic_node("exc_thalamus");
    structure->connect_layers_internal(node2, "input_layer2",
        ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(5)));
    structure->connect_layers("input_layer2", "inh_thalamus2",
        ConnectionConfig(false, 0, 10, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(10)));
    structure->connect_layers("exc_cortex", "inh_thalamus2",
        ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(7,1)));
    structure->connect_layers_internal(node2, "inh_thalamus2",
        ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            ArborizedConfig(5,1)));

    structure->connect_layers_matching("exc_cortex",
        LayerConfig("output_layer", neural_model, "low_threshold"),
        ConnectionConfig(true, 40, 0.1, CONVERGENT, ADD,
            new RandomWeightConfig(0.025),
            ArborizedConfig(15,1)));
    structure->connect_layers("output_layer", "exc_cortex",
        ConnectionConfig(false, 40, 1, CONVERGENT, ADD,
            new RandomWeightConfig(0.5),
            ArborizedConfig(15,1)));

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
    Structure *structure = model->add_structure("hh");

    int resolution = 128;
    structure->add_layer(LayerConfig("input_layer",
        HODGKIN_HUXLEY, 1, 10, "0"));
    structure->add_layer(LayerConfig("exc_thalamus",
        HODGKIN_HUXLEY, resolution, resolution, "0", 0.5));
    structure->add_layer(LayerConfig("inh_thalamus",
        HODGKIN_HUXLEY, resolution, resolution, "0"));
    structure->add_layer(LayerConfig("exc_cortex",
        HODGKIN_HUXLEY, resolution, resolution, "0"));
    structure->add_layer(LayerConfig("inh_cortex",
        HODGKIN_HUXLEY, resolution, resolution, "0"));

    structure->connect_layers("input_layer", "exc_thalamus",
        ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(5)));
    structure->connect_layers("exc_thalamus", "exc_cortex",
        ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(7,1)));
    structure->connect_layers("exc_cortex", "inh_cortex",
        ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(9,1)));
    structure->connect_layers("exc_cortex", "exc_cortex",
        ConnectionConfig(true, 2, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(5,1)));
    structure->connect_layers("inh_cortex", "exc_cortex",
        ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            ArborizedConfig(5,1)));
    structure->connect_layers("exc_cortex", "inh_thalamus",
        ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            ArborizedConfig(7,1)));
    structure->connect_layers("inh_thalamus", "exc_thalamus",
        ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            ArborizedConfig(5,1)));

    structure->connect_layers_matching("exc_cortex",
        LayerConfig("output_layer", HODGKIN_HUXLEY, "0"),
        ConnectionConfig(true, 40, 0.1, CONVERGENT, ADD,
            new RandomWeightConfig(0.025),
            ArborizedConfig(15,1)));
        //ConnectionConfig(true, 40, 0.1, CONVERGENT, ADD,
        //    RandomWeightConfig(0.0001),
        //    ArborizedConfig(15,1)));
    structure->connect_layers("output_layer", "exc_cortex",
        ConnectionConfig(false, 40, 1, CONVERGENT, ADD,
            new RandomWeightConfig(0.5),
            ArborizedConfig(15,1)));

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

Model* build_cc_model(NeuralModel neural_model) {
    /* Construct the model */
    Model *model = new Model();
    std::vector<Structure*> structures;
    int num_structures = 4;

    for (int i = 0 ; i < num_structures ; ++i) {
        Structure *structure = model->add_structure(std::to_string(i));

        int resolution = 128;
        structure->add_layer(LayerConfig("input_layer",
            neural_model, 1, 10, "default"));
        structure->add_layer(LayerConfig("exc_thalamus",
            neural_model, resolution, resolution, "low_threshold", 0.5));
        structure->add_layer(LayerConfig("inh_thalamus",
            neural_model, resolution, resolution, "default"));
        structure->add_layer(LayerConfig("exc_cortex",
            neural_model, resolution, resolution, "thalamo_cortical"));
        structure->add_layer(LayerConfig("inh_cortex",
            neural_model, resolution, resolution, "default"));

        structure->connect_layers("input_layer", "exc_thalamus",
            ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
                new RandomWeightConfig(5)));
        structure->connect_layers("exc_thalamus", "exc_cortex",
            ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
                new RandomWeightConfig(0.25),
                ArborizedConfig(7,1)));
        structure->connect_layers("exc_cortex", "inh_cortex",
            ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
                new RandomWeightConfig(0.25),
                ArborizedConfig(9,1)));
        structure->connect_layers("exc_cortex", "exc_cortex",
            ConnectionConfig(true, 2, 5, CONVERGENT, ADD,
                new RandomWeightConfig(0.25),
                ArborizedConfig(5,1)));
        structure->connect_layers("inh_cortex", "exc_cortex",
            ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
                new RandomWeightConfig(10),
                ArborizedConfig(5,1)));
        structure->connect_layers("exc_cortex", "inh_thalamus",
            ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
                new RandomWeightConfig(0.25),
                ArborizedConfig(7,1)));
        structure->connect_layers("inh_thalamus", "exc_thalamus",
            ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
                new RandomWeightConfig(10),
                ArborizedConfig(5,1)));

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
            //ConnectionConfig(true, 20, 1, ONE_TO_ONE, ADD,
            //    new RandomWeightConfig(0.1)));
            ConnectionConfig(true, 10, 0.1, CONVERGENT, MULT,
                new RandomWeightConfig(0.01),
                ArborizedConfig(9,1)));
            //ConnectionConfig(true, 20, 0.01, FULLY_CONNECTED, ADD,
            //    new RandomWeightConfig(0.001)));
    }

    return model;
}

Model* build_re_model() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("re", FEEDFORWARD);

    int resolution = 128;
    structure->add_layer(LayerConfig("in", HEBBIAN_RATE_ENCODING, 1, 8));
    structure->add_layer(LayerConfig("hid", HEBBIAN_RATE_ENCODING, resolution, resolution));
    structure->add_layer(LayerConfig("out", HEBBIAN_RATE_ENCODING, resolution, resolution));

    structure->connect_layers("in", "hid",
        ConnectionConfig(true, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(5)));
    structure->connect_layers("hid", "out",
        ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(5),
            ArborizedConfig(9,1)));

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
    model = build_stress_model(IZHIKEVICH);
    print_model(model);
    run_simulation(model, 10, true);
    std::cout << "\n";

    delete model;
}

void reentrant_image_test() {
    Model *model;

    std::cout << "Reentrant Image...\n";
    model = build_reentrant_image_model(IZHIKEVICH);
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
    model = build_image_model(IZHIKEVICH);
    //model = build_image_model(HEBBIAN_RATE_ENCODING);
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
    model = build_alignment_model(IZHIKEVICH);
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
    model = build_dendritic_model(IZHIKEVICH);
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
    //model = build_alignment_model(HODGKIN_HUXLEY);
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
    model = build_cc_model(IZHIKEVICH);
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

void mnist_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("mnist");

    int resolution = 1024;
    structure->add_layer(LayerConfig("input_layer",
        IZHIKEVICH, 28, 28, "default"));

    int num_hidden = 10;
    for (int i = 0; i < num_hidden; ++i) {
        structure->add_layer(LayerConfig(std::to_string(i),
            IZHIKEVICH, 28, 28, "default", 0.5));
        structure->connect_layers("input_layer", std::to_string(i),
            ConnectionConfig(true, 0, 5, CONVOLUTIONAL, ADD,
                new RandomWeightConfig(5),
                ArborizedConfig(5,1)));

        structure->connect_layers(std::to_string(i), std::to_string(i),
            ConnectionConfig(true, 0, 1, CONVOLUTIONAL, ADD,
                new RandomWeightConfig(0.1),
                ArborizedConfig(5,1)));
        structure->connect_layers(std::to_string(i), std::to_string(i),
            ConnectionConfig(false, 0, 2, CONVOLUTIONAL, DIV,
                new RandomWeightConfig(2),
                ArborizedConfig(7,1)));
    }

    for (int i = 0; i < num_hidden; ++i)
        for (int j = 0; j < num_hidden; ++j)
            if (i != j)
                structure->connect_layers(std::to_string(i), std::to_string(j),
                    ConnectionConfig(false, 0, 5, ONE_TO_ONE, DIV,
                        new RandomWeightConfig(1)));

    // Modules
    std::string output_name = "visualizer_output";

    structure->add_module("input_layer", "csv_input", "/HDD/datasets/mnist/mnist_test.csv 1 5000 25");
    structure->add_module("input_layer", output_name);
    for (int i = 0; i < num_hidden; ++i)
        structure->add_module(std::to_string(i), output_name);

    std::cout << "CSV test......\n";
    print_model(model);
    run_simulation(model, 100000, true);
    std::cout << "\n";

    delete model;
}

void divergent_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("divergent");

    structure->add_layer(LayerConfig("in", IZHIKEVICH, 10, 10, "default"));
    structure->add_layer(LayerConfig("out", IZHIKEVICH, 100, 55, "default"));

    structure->connect_layers("in", "out",
        ConnectionConfig(true, 0, 100, DIVERGENT, ADD,
            new RandomWeightConfig(100),
            ArborizedConfig(10,10,10,5)));

    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";
    //std::string output_name = "print_output";

    structure->add_module("in", "random_input", "10 5000");
    structure->add_module("in", output_name, "8");
    structure->add_module("out", output_name, "8");

    std::cout << "Divergent test......\n";
    print_model(model);
    run_simulation(model, 100000, true);
    std::cout << "\n";

    delete model;
}

void speech_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("speech", PARALLEL);

    // Input layer
    structure->add_layer(LayerConfig("input_layer",
        IZHIKEVICH, 1, 41, "thalamo_cortical"));

    // Convergent layer
    structure->connect_layers_expected("input_layer",
        LayerConfig("convergent_layer", IZHIKEVICH, "regular"),
        ConnectionConfig(true, 0, 100, CONVERGENT, ADD,
            new RandomWeightConfig(10),
            ArborizedConfig(1,3,1,1)));

    int spread = 1;
    int vertical_inhibition_spread = 1;

    // Vertical cluster layer
    structure->connect_layers_expected("convergent_layer",
        LayerConfig("vertical_layer", IZHIKEVICH, "bursting"), //, 0.5),
        ConnectionConfig(true, 0, 10, DIVERGENT, ADD,
            new RandomWeightConfig(0.5),
            ArborizedConfig(spread,10,1,1)));
    // Vertical cluster inhibitory layer
    structure->connect_layers_expected("vertical_layer",
        LayerConfig("vertical_inhibitory", IZHIKEVICH, "fast"),
        ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(0.5),
            ArborizedConfig(vertical_inhibition_spread,1,1,1)));
    structure->connect_layers("vertical_inhibitory", "vertical_layer",
        ConnectionConfig(false, 0, 5, DIVERGENT, DIV,
            new RandomWeightConfig(5),
            ArborizedConfig(vertical_inhibition_spread,1,1,1)));

    // Block cluster layer
    structure->connect_layers_expected("vertical_layer",
        LayerConfig("cluster_layer", IZHIKEVICH, "bursting", 0.5),
        ConnectionConfig(true, 0, 10, DIVERGENT, ADD,
            new RandomWeightConfig(0.5),
            ArborizedConfig(spread,5,1,1)));
    // Block cluster inhibitory layer
    structure->connect_layers_expected("cluster_layer",
        LayerConfig("cluster_inhibitory", IZHIKEVICH, "fast"),
        ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(0.5),
            ArborizedConfig(spread,5,1,1)));
    structure->connect_layers("cluster_inhibitory", "cluster_layer",
        ConnectionConfig(false, 0, 5, DIVERGENT, DIV,
            new RandomWeightConfig(5),
            ArborizedConfig(spread,10,1,1)));

    int mot_f_size = 3;
    int mot_stride = 1;
    int offset = mot_f_size;
    int pool_f_size = 10;
    int pool_stride = 3;
    float pos_strength = 5;
    float neg_strength = 1;
    float mid_strength = 10;
    float pool_strength = 5;

    // Motion detectors
    structure->connect_layers_expected("cluster_layer",
        LayerConfig("motion_up", IZHIKEVICH, "bursting", 1),
        ConnectionConfig(false, 5, 10, CONVERGENT, ADD,
            new RandomWeightConfig(pos_strength),
            ArborizedConfig(1,mot_f_size,1,mot_stride,offset,offset)));
    structure->connect_layers("cluster_layer", "motion_up",
        ConnectionConfig(false, 5, 10, CONVERGENT, DIV,
            new RandomWeightConfig(neg_strength),
            ArborizedConfig(1,mot_f_size,1,mot_stride,-offset,-offset)));
    structure->connect_layers("cluster_layer", "motion_up",
        ConnectionConfig(false, 0, 10, CONVERGENT, MULT,
            new RandomWeightConfig(mid_strength),
            ArborizedConfig(1,mot_f_size,1,mot_stride)));
    structure->connect_layers_expected("motion_up",
        LayerConfig("motion_up_pool", IZHIKEVICH, "chattering", 1),
        ConnectionConfig(false, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(pool_strength),
            ArborizedConfig(1,pool_f_size,1,pool_stride)));

    structure->connect_layers_expected("cluster_layer",
        LayerConfig("motion_down", IZHIKEVICH, "bursting", 1),
        ConnectionConfig(false, 5, 10, CONVERGENT, ADD,
            new RandomWeightConfig(pos_strength),
            ArborizedConfig(1,mot_f_size,1,mot_stride,-offset,-offset)));
    structure->connect_layers("cluster_layer", "motion_down",
        ConnectionConfig(false, 5, 10, CONVERGENT, DIV,
            new RandomWeightConfig(neg_strength),
            ArborizedConfig(1,mot_f_size,1,mot_stride,offset,offset)));
    structure->connect_layers("cluster_layer", "motion_down",
        ConnectionConfig(false, 0, 10, CONVERGENT, MULT,
            new RandomWeightConfig(mid_strength),
            ArborizedConfig(1,mot_f_size,1,mot_stride)));
    structure->connect_layers_expected("motion_down",
        LayerConfig("motion_down_pool", IZHIKEVICH, "chattering", 1),
        ConnectionConfig(false, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(pool_strength),
            ArborizedConfig(1,pool_f_size,1,pool_stride)));

    // Modules
    std::string output_name = "visualizer_output";

    //structure->add_module("input_layer", "random_input", "10 500");
    //structure->add_module("input_layer", "csv_input", "./resources/english.csv 0 1 0.1");
    structure->add_module("input_layer", "csv_input", "./resources/substitute.csv 0 1 0.01");

    //structure->add_module("input_layer", output_name);
    //structure->add_module("convergent_layer", output_name);
    //structure->add_module("vertical_layer", output_name);
    structure->add_module("cluster_layer", output_name);
    structure->add_module("motion_up", output_name);
    structure->add_module("motion_up_pool", output_name);
    structure->add_module("motion_down", output_name);
    structure->add_module("motion_down_pool", output_name);

    std::cout << "Speech test......\n";
    print_model(model);
    Clock clock((float)120.0);
    clock.run(model, 1000000, true);
    std::cout << "\n";

    delete model;
}

void second_order_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("second_order");

    structure->add_layer(LayerConfig("in1", IZHIKEVICH, 100, 100, "default"));
    structure->add_layer(LayerConfig("in2", IZHIKEVICH, 100, 100, "default"));
    structure->add_layer(LayerConfig("out", IZHIKEVICH, 100, 100, "default"));

    structure->get_dendritic_root("out")->set_second_order();

    structure->connect_layers("in1", "out",
        ConnectionConfig(false, 0, 100, ONE_TO_ONE, ADD,
            new RandomWeightConfig(10)));
    structure->connect_layers("in2", "out",
        ConnectionConfig(false, 0, 100, ONE_TO_ONE, MULT,
            new RandomWeightConfig(10)));

    // Modules
    std::string output_name = "visualizer_output";
    //std::string output_name = "dummy_output";

    structure->add_module("in1", "random_input", "10 5000");
    structure->add_module("in2", "random_input", "10 5000");
    structure->add_module("in1", output_name);
    structure->add_module("in2", output_name);
    structure->add_module("out", output_name);

    std::cout << "Second order test......\n";
    print_model(model);
    run_simulation(model, 100000, true);
    std::cout << "\n";

    delete model;
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(nullptr));

    try {
        //stress_test();
        //image_test();
        //reentrant_image_test();
        alignment_test();
        //dendritic_test();
        //hh_test();
        //cc_test();
        //re_test();
        //mnist_test();
        //divergent_test();
        //second_order_test();
        //speech_test();

        return 0;
    } catch (const char* msg) {
        printf("\n\nEXPECTED: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
