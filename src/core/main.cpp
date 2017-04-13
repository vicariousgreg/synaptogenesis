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
    structure->add_layer(new LayerConfig("pos",
        neural_model, 1, size, "random positive", 5));
    structure->add_layer(new LayerConfig("neg",
        neural_model, 1, size / 4, "random negative", 2));
    structure->connect_layers("pos", "pos",
        new ConnectionConfig(false, 0, .5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(0.5)));
    structure->connect_layers("pos", "neg",
        new ConnectionConfig(false, 0, .5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(0.5)));
    structure->connect_layers("neg", "pos",
        new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, SUB,
            new RandomWeightConfig(1)));
    structure->connect_layers("neg", "neg",
        new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, SUB,
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
        new LayerConfig("photoreceptor", neural_model, "default"));

    // Vertical line detection
    structure->connect_layers_expected("photoreceptor",
        new LayerConfig("vertical", neural_model, "default"),
        new ConnectionConfig(
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
            new ArborizedConfig(11,1)));
    structure->connect_layers("vertical", "vertical",
        new ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                "-5  0  5  0 -5 "
                "-5  0  5  0 -5 "
                "-5  0  5  0 -5 "
                "-5  0  5  0 -5 "
                "-5  0  5  0 -5 "),
            new ArborizedConfig(5,1)));

    // Horizontal line detection
    structure->connect_layers_expected("photoreceptor",
        new LayerConfig("horizontal", neural_model, "default"),
        new ConnectionConfig(
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
            new ArborizedConfig(11,1)));
    structure->connect_layers("horizontal", "horizontal",
        new ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                "-5 -5 -5 -5 -5 "
                " 0  0  0  0  0 "
                " 5  5  5  5  5 "
                " 0  0  0  0  0 "
                "-5 -5 -5 -5 -5 "),
            new ArborizedConfig(5,1)));

    // Cross detection
    structure->connect_layers_expected("vertical",
        new LayerConfig("cross", neural_model, "default"),
        new ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                "-5  0 10  0 -5 "
                "-5  0 10  0 -5 "
                "-5  0 10  0 -5 "
                "-5  0 10  0 -5 "
                "-5  0 10  0 -5 "),
            new ArborizedConfig(5,1)));
    structure->connect_layers("horizontal", "cross",
        new ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                "-5 -5 -5 -5 -5 "
                " 0  0  0  0  0 "
                "10 10 10 10 10 "
                " 0  0  0  0  0 "
                "-5 -5 -5 -5 -5 "),
            new ArborizedConfig(5,1)));

    // Forward slash
    structure->connect_layers_expected("photoreceptor",
        new LayerConfig("forward_slash", neural_model, "default"),
        new ConnectionConfig(
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
            new ArborizedConfig(9,1)));
    structure->connect_layers("forward_slash", "forward_slash",
        new ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                " 0  0 -5  0  5 "
                " 0 -5  0  5  0 "
                "-5  0  5  0 -5 "
                " 0  5  0 -5  0 "
                " 5  0 -5  0  0 "),
            new ArborizedConfig(5,1)));

    // Back slash
    structure->connect_layers_expected("photoreceptor",
        new LayerConfig("back_slash", neural_model, "default"),
            new ConnectionConfig(
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
            new ArborizedConfig(9,1)));
    structure->connect_layers("back_slash", "back_slash",
        new ConnectionConfig(
            false, 0, 5, CONVOLUTIONAL, ADD,
            new SpecifiedWeightConfig(
                " 5  0 -5  0  0 "
                " 0  5  0 -5  0 "
                "-5  0  5  0 -5 "
                " 0 -5  0  5  0 "
                " 0  0 -5  0  5 "),
            new ArborizedConfig(5,1)));

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
        new LayerConfig("photoreceptor", neural_model, "default"));

    // Connect first layer to receptor
    structure->connect_layers_matching("photoreceptor",
        new LayerConfig("layer1", neural_model, "default"),
        new ConnectionConfig(false, 0, 1, CONVOLUTIONAL, ADD,
            new RandomWeightConfig(1),
            new ArborizedConfig(21,1)));

    // Create reentrant pair
    structure->connect_layers_matching("layer1",
        new LayerConfig("layer2", neural_model, "default", 5),
        new ConnectionConfig(false, 0, 1, CONVOLUTIONAL, ADD,
            new RandomWeightConfig(1),
            new ArborizedConfig(9,1)));
    structure->connect_layers("layer2","layer1",
        new ConnectionConfig(false, 0, 1, CONVOLUTIONAL, ADD,
            new RandomWeightConfig(1),
            new ArborizedConfig(9,1)));

    // Inhibitory self connections
    structure->connect_layers("layer1", "layer1",
        new ConnectionConfig(false, 0, 1, CONVOLUTIONAL, SUB,
            new RandomWeightConfig(10),
            new ArborizedConfig(5,1)));

    structure->connect_layers("layer2", "layer2",
        new ConnectionConfig(false, 0, 1, CONVOLUTIONAL, SUB,
            new RandomWeightConfig(10),
            new ArborizedConfig(5,1)));

    // Modules
    structure->add_module("photoreceptor", "image_input", image_path);
    structure->add_module("photoreceptor", output_name, "8");
    structure->add_module("layer1", output_name, "8");
    structure->add_module("layer2", output_name, "8");

    return model;
}

Model* build_working_memory_model(NeuralModel neural_model) {
    /* Construct the model */
    Model *model = new Model();
    Structure *main_structure = model->add_structure("working memory", PARALLEL);

    int thal_ratio = 1;
    int cortex_size = 128;
    int thal_size = cortex_size / thal_ratio;

    float ff_noise = 0.0;
    float thal_noise = 0.0;
    float cortex_noise = 1.0;

    float ampa = 3;
    float nmda = 10;
    float gabaa = 3;
    float gabab = 10;

    bool exc_plastic = true;
    bool inh_plastic = false;

    int sensory_center = 15;
    int sensory_surround = 15;
    int inter_cortex_center = 9;
    int inter_cortex_surround = 15;
    int gamma_center = 3;
    int gamma_surround = 3;

    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    // Feedforward circuit
    main_structure->add_layer(new LayerConfig("input_layer", RELAY, 1, 10));
    main_structure->add_layer(new LayerConfig("feedforward",
        neural_model, cortex_size, cortex_size, "regular", ff_noise));

    // Thalamic relay
    main_structure->add_layer(new LayerConfig("tl1_thalamus",
        neural_model, 1, 1, "thalamo_cortical", thal_noise));

    // Feedforward input
    main_structure->connect_layers("input_layer", "feedforward",
        new ConnectionConfig(false, 0, 5, FULLY_CONNECTED, AMPA,
            new RandomWeightConfig(100, 0.1)));

    std::vector<Structure*> sub_structures;
    for (int i = 0 ; i < 2 ; ++i) {
        Structure *sub_structure =
            model->add_structure(std::to_string(i), PARALLEL);
        sub_structures.push_back(sub_structure);

        // Thalamic gamma nucleus
        sub_structure->add_layer(new LayerConfig("gamma_thalamus",
            neural_model, thal_size, thal_size, "thalamo_cortical", thal_noise));

        // Cortical layers
        sub_structure->add_layer(new LayerConfig("3_cortex",
            neural_model, cortex_size, cortex_size, "random positive", cortex_noise));
        sub_structure->add_layer(new LayerConfig("6_cortex",
            neural_model, cortex_size, cortex_size, "regular", cortex_noise));

        // Cortico-cortical connectivity
        sub_structure->connect_layers("3_cortex", "6_cortex",
            new ConnectionConfig(exc_plastic, 0, 4, CONVERGENT, NMDA,
                new FlatWeightConfig(nmda),
                new ArborizedConfig(inter_cortex_center,1)));

        sub_structure->connect_layers("3_cortex", "6_cortex",
            new ConnectionConfig(exc_plastic, 0, 4, CONVERGENT, AMPA,
                new FlatWeightConfig(ampa/3),
                new ArborizedConfig(inter_cortex_center,1)));

        sub_structure->connect_layers("6_cortex", "3_cortex",
            new ConnectionConfig(exc_plastic, 0, 4, CONVERGENT, NMDA,
                new FlatWeightConfig(nmda),
                new ArborizedConfig(inter_cortex_center,1)));

        sub_structure->connect_layers("6_cortex", "3_cortex",
            new ConnectionConfig(exc_plastic, 0, 4, CONVERGENT, AMPA,
                new FlatWeightConfig(ampa/3),
                new ArborizedConfig(inter_cortex_center,1)));

        sub_structure->connect_layers("3_cortex", "6_cortex",
            new ConnectionConfig(inh_plastic, 0, 4, CONVERGENT, GABAB,
                new SurroundWeightConfig(inter_cortex_center,
                    new FlatWeightConfig(gabab)),
                new ArborizedConfig(inter_cortex_surround,1)));
        sub_structure->connect_layers("6_cortex", "3_cortex",
            new ConnectionConfig(inh_plastic, 0, 4, CONVERGENT, GABAB,
                new SurroundWeightConfig(inter_cortex_center,
                    new FlatWeightConfig(gabab)),
                new ArborizedConfig(inter_cortex_surround,1)));

        // Gamma connectivity
        sub_structure->connect_layers("gamma_thalamus", "3_cortex",
            new ConnectionConfig(exc_plastic, 10, 4, CONVERGENT, AMPA,
                new FlatWeightConfig(ampa*thal_ratio),
                new ArborizedConfig(gamma_center,1)));
        sub_structure->connect_layers("gamma_thalamus", "3_cortex",
            new ConnectionConfig(inh_plastic, 10, 4, CONVERGENT, GABAA,
                new SurroundWeightConfig(gamma_center,
                    new FlatWeightConfig(gabaa*thal_ratio)),
                new ArborizedConfig(gamma_surround,1)));
        sub_structure->connect_layers("6_cortex", "gamma_thalamus",
            new ConnectionConfig(exc_plastic, 10, 4, CONVERGENT, AMPA,
                new FlatWeightConfig(ampa*thal_ratio),
                new ArborizedConfig(gamma_center,1)));
        sub_structure->connect_layers("6_cortex", "gamma_thalamus",
            new ConnectionConfig(inh_plastic, 10, 4, CONVERGENT, GABAA,
                new SurroundWeightConfig(gamma_center,
                    new FlatWeightConfig(gabaa*thal_ratio)),
                new ArborizedConfig(gamma_surround,1)));

        // Thalamocortical control connectivity
        Structure::connect(main_structure, "tl1_thalamus",
            sub_structure, "3_cortex",
            new ConnectionConfig(false, 0, 4, FULLY_CONNECTED, MULT,
                new FlatWeightConfig(ampa*thal_ratio)));
        Structure::connect(main_structure, "tl1_thalamus",
            sub_structure, "6_cortex",
            new ConnectionConfig(false, 0, 4, FULLY_CONNECTED, MULT,
                new FlatWeightConfig(ampa*thal_ratio)));

        sub_structure->add_module("3_cortex", output_name, "8");
        sub_structure->add_module("6_cortex", output_name, "8");
        sub_structure->add_module("gamma_thalamus", output_name, "8");
    }

    // Sensory relay to cortex
    for (int i = 0 ; i < 1 ; ++i) {
        Structure::connect(main_structure, "feedforward",
            sub_structures[i], "3_cortex",
            new ConnectionConfig(exc_plastic, 0, 4, CONVERGENT, AMPA,
                new FlatWeightConfig(ampa*thal_ratio),
                new ArborizedConfig(sensory_center,1)));
        Structure::connect(main_structure, "feedforward",
            sub_structures[i], "3_cortex",
            new ConnectionConfig(inh_plastic, 0, 4, CONVERGENT, GABAA,
                new SurroundWeightConfig(sensory_center,
                    new FlatWeightConfig(gabaa*thal_ratio)),
                new ArborizedConfig(sensory_surround,1)));
    }

    Structure::connect(sub_structures[0], "3_cortex",
        sub_structures[1], "3_cortex",
        new ConnectionConfig(exc_plastic, 0, 4, CONVERGENT, AMPA,
            new FlatWeightConfig(ampa*thal_ratio),
            new ArborizedConfig(sensory_center,1)));
    Structure::connect(sub_structures[0], "3_cortex",
        sub_structures[1], "3_cortex",
        new ConnectionConfig(inh_plastic, 0, 4, CONVERGENT, GABAA,
            new SurroundWeightConfig(sensory_center,
                new FlatWeightConfig(gabaa*thal_ratio)),
            new ArborizedConfig(sensory_surround,1)));

    // Modules
    main_structure->add_module("input_layer", "one_hot_random_input", "1 5000");
    main_structure->add_module("tl1_thalamus", "random_input", "3 500");
    main_structure->add_module("feedforward", output_name, "8");

    return model;
}

Model* build_dendritic_model(NeuralModel neural_model) {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("dendritic");

    int resolution = 128;
    structure->add_layer(new LayerConfig("input_layer1",
        neural_model, 1, 10, "default"));
    structure->add_layer(new LayerConfig("input_layer2",
        neural_model, 1, 10, "default"));
    structure->add_layer(new LayerConfig("exc_thalamus",
        neural_model, resolution, resolution, "low_threshold", 0.5));
    structure->add_layer(new LayerConfig("inh_thalamus1",
        neural_model, resolution, resolution, "default"));
    structure->add_layer(new LayerConfig("inh_thalamus2",
        neural_model, resolution, resolution, "default"));
    structure->add_layer(new LayerConfig("exc_cortex",
        neural_model, resolution, resolution, "thalamo_cortical"));
    structure->add_layer(new LayerConfig("inh_cortex",
        neural_model, resolution, resolution, "default"));

    structure->connect_layers("exc_thalamus", "exc_cortex",
        new ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            new ArborizedConfig(7,1)));
    structure->connect_layers("exc_cortex", "inh_cortex",
        new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            new ArborizedConfig(9,1)));
    structure->connect_layers("exc_cortex", "exc_cortex",
        new ConnectionConfig(true, 2, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            new ArborizedConfig(5,1)));
    structure->connect_layers("inh_cortex", "exc_cortex",
        new ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            new ArborizedConfig(5,1)));

    // Input branch 1
    auto node1 = structure->spawn_dendritic_node("exc_thalamus");
    structure->connect_layers_internal(node1, "input_layer1",
        new ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(5)));
    structure->connect_layers("input_layer1", "inh_thalamus1",
        new ConnectionConfig(false, 0, 10, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(10)));
    structure->connect_layers("exc_cortex", "inh_thalamus1",
        new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            new ArborizedConfig(7,1)));
    structure->connect_layers_internal(node1, "inh_thalamus1",
        new ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            new ArborizedConfig(5,1)));

    // Input branch 2
    auto node2 = structure->spawn_dendritic_node("exc_thalamus");
    structure->connect_layers_internal(node2, "input_layer2",
        new ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(5)));
    structure->connect_layers("input_layer2", "inh_thalamus2",
        new ConnectionConfig(false, 0, 10, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(10)));
    structure->connect_layers("exc_cortex", "inh_thalamus2",
        new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            new ArborizedConfig(7,1)));
    structure->connect_layers_internal(node2, "inh_thalamus2",
        new ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            new ArborizedConfig(5,1)));

    structure->connect_layers_matching("exc_cortex",
        new LayerConfig("output_layer", neural_model, "low_threshold"),
        new ConnectionConfig(true, 40, 0.1, CONVERGENT, ADD,
            new RandomWeightConfig(0.025),
            new ArborizedConfig(15,1)));
    structure->connect_layers("output_layer", "exc_cortex",
        new ConnectionConfig(false, 40, 1, CONVERGENT, ADD,
            new RandomWeightConfig(0.5),
            new ArborizedConfig(15,1)));

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
    structure->add_layer(new LayerConfig("input_layer",
        HODGKIN_HUXLEY, 1, 10, "0"));
    structure->add_layer(new LayerConfig("exc_thalamus",
        HODGKIN_HUXLEY, resolution, resolution, "0", 0.5));
    structure->add_layer(new LayerConfig("inh_thalamus",
        HODGKIN_HUXLEY, resolution, resolution, "0"));
    structure->add_layer(new LayerConfig("exc_cortex",
        HODGKIN_HUXLEY, resolution, resolution, "0"));
    structure->add_layer(new LayerConfig("inh_cortex",
        HODGKIN_HUXLEY, resolution, resolution, "0"));

    structure->connect_layers("input_layer", "exc_thalamus",
        new ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(5)));
    structure->connect_layers("exc_thalamus", "exc_cortex",
        new ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            new ArborizedConfig(7,1)));
    structure->connect_layers("exc_cortex", "inh_cortex",
        new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            new ArborizedConfig(9,1)));
    structure->connect_layers("exc_cortex", "exc_cortex",
        new ConnectionConfig(true, 2, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            new ArborizedConfig(5,1)));
    structure->connect_layers("inh_cortex", "exc_cortex",
        new ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            new ArborizedConfig(5,1)));
    structure->connect_layers("exc_cortex", "inh_thalamus",
        new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.25),
            new ArborizedConfig(7,1)));
    structure->connect_layers("inh_thalamus", "exc_thalamus",
        new ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
            new RandomWeightConfig(5),
            new ArborizedConfig(5,1)));

    structure->connect_layers_matching("exc_cortex",
        new LayerConfig("output_layer", HODGKIN_HUXLEY, "0"),
        new ConnectionConfig(true, 40, 0.1, CONVERGENT, ADD,
            new RandomWeightConfig(0.025),
            new ArborizedConfig(15,1)));
        //new ConnectionConfig(true, 40, 0.1, CONVERGENT, ADD,
        //    RandomWeightConfig(0.0001),
        //    new ArborizedConfig(15,1)));
    structure->connect_layers("output_layer", "exc_cortex",
        new ConnectionConfig(false, 40, 1, CONVERGENT, ADD,
            new RandomWeightConfig(0.5),
            new ArborizedConfig(15,1)));

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
        structure->add_layer(new LayerConfig("input_layer",
            neural_model, 1, 10, "default"));
        structure->add_layer(new LayerConfig("exc_thalamus",
            neural_model, resolution, resolution, "low_threshold", 0.5));
        structure->add_layer(new LayerConfig("inh_thalamus",
            neural_model, resolution, resolution, "default"));
        structure->add_layer(new LayerConfig("exc_cortex",
            neural_model, resolution, resolution, "thalamo_cortical"));
        structure->add_layer(new LayerConfig("inh_cortex",
            neural_model, resolution, resolution, "default"));

        structure->connect_layers("input_layer", "exc_thalamus",
            new ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
                new RandomWeightConfig(5)));
        structure->connect_layers("exc_thalamus", "exc_cortex",
            new ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
                new RandomWeightConfig(0.25),
                new ArborizedConfig(7,1)));
        structure->connect_layers("exc_cortex", "inh_cortex",
            new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
                new RandomWeightConfig(0.25),
                new ArborizedConfig(9,1)));
        structure->connect_layers("exc_cortex", "exc_cortex",
            new ConnectionConfig(true, 2, 5, CONVERGENT, ADD,
                new RandomWeightConfig(0.25),
                new ArborizedConfig(5,1)));
        structure->connect_layers("inh_cortex", "exc_cortex",
            new ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
                new RandomWeightConfig(10),
                new ArborizedConfig(5,1)));
        structure->connect_layers("exc_cortex", "inh_thalamus",
            new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
                new RandomWeightConfig(0.25),
                new ArborizedConfig(7,1)));
        structure->connect_layers("inh_thalamus", "exc_thalamus",
            new ConnectionConfig(false, 0, 5, CONVERGENT, DIV,
                new RandomWeightConfig(10),
                new ArborizedConfig(5,1)));

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
            //new ConnectionConfig(true, 20, 1, ONE_TO_ONE, ADD,
            //    new RandomWeightConfig(0.1)));
            new ConnectionConfig(true, 10, 0.1, CONVERGENT, MULT,
                new RandomWeightConfig(0.01),
                new ArborizedConfig(9,1)));
            //new ConnectionConfig(true, 20, 0.01, FULLY_CONNECTED, ADD,
            //    new RandomWeightConfig(0.001)));
    }

    return model;
}

Model* build_re_model() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("re", FEEDFORWARD);

    int resolution = 128;
    structure->add_layer(new LayerConfig("in", HEBBIAN_RATE_ENCODING, 1, 8));
    structure->add_layer(new LayerConfig("hid", HEBBIAN_RATE_ENCODING, resolution, resolution));
    structure->add_layer(new LayerConfig("out", HEBBIAN_RATE_ENCODING, resolution, resolution));

    structure->connect_layers("in", "hid",
        new ConnectionConfig(true, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(5)));
    structure->connect_layers("hid", "out",
        new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(5),
            new ArborizedConfig(9,1)));

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

void working_memory_test() {
    Model *model;

    std::cout << "Alignment...\n";
    model = build_working_memory_model(IZHIKEVICH);
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
    structure->add_layer(new LayerConfig("input_layer",
        IZHIKEVICH, 28, 28, "default"));

    int num_hidden = 10;
    for (int i = 0; i < num_hidden; ++i) {
        structure->add_layer(new LayerConfig(std::to_string(i),
            IZHIKEVICH, 28, 28, "default", 0.5));
        structure->connect_layers("input_layer", std::to_string(i),
            new ConnectionConfig(true, 0, 5, CONVOLUTIONAL, ADD,
                new RandomWeightConfig(5),
                new ArborizedConfig(5,1)));

        structure->connect_layers(std::to_string(i), std::to_string(i),
            new ConnectionConfig(true, 0, 1, CONVOLUTIONAL, ADD,
                new RandomWeightConfig(0.1),
                new ArborizedConfig(5,1)));
        structure->connect_layers(std::to_string(i), std::to_string(i),
            new ConnectionConfig(false, 0, 2, CONVOLUTIONAL, DIV,
                new RandomWeightConfig(2),
                new ArborizedConfig(7,1)));
    }

    for (int i = 0; i < num_hidden; ++i)
        for (int j = 0; j < num_hidden; ++j)
            if (i != j)
                structure->connect_layers(std::to_string(i), std::to_string(j),
                    new ConnectionConfig(false, 0, 5, ONE_TO_ONE, DIV,
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

    structure->add_layer(new LayerConfig("in", IZHIKEVICH, 10, 10, "default"));
    structure->add_layer(new LayerConfig("out", IZHIKEVICH, 100, 55, "default"));

    structure->connect_layers("in", "out",
        new ConnectionConfig(true, 0, 100, DIVERGENT, ADD,
            new RandomWeightConfig(100),
            new ArborizedConfig(10,10,10,5)));

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
    structure->add_layer(new LayerConfig("input_layer",
        IZHIKEVICH, 1, 41, "fast"));

    // Convergent layers
    structure->connect_layers_matching("input_layer",
        new LayerConfig("convergent_layer", IZHIKEVICH, "fast"),
        new ConnectionConfig(false, 2, 100, CONVERGENT, ADD,
            new FlatWeightConfig(10),
            new ArborizedConfig(1,3,1,1)));

    structure->connect_layers_matching("input_layer",
        new LayerConfig("convergent_layer_inhibitory", IZHIKEVICH, "fast"),
        new ConnectionConfig(false, 0, 100, CONVERGENT, ADD,
            new FlatWeightConfig(3),
            new ArborizedConfig(1,10,1,1)));
    structure->connect_layers("convergent_layer_inhibitory",
        "convergent_layer",
        new ConnectionConfig(false, 0, 100, ONE_TO_ONE, SUB,
            new FlatWeightConfig(10)));

    /*
    int vertical_spread = 1;
    int horizontal_spread = 10;

    // Vertical cluster layer
    structure->connect_layers_expected("convergent_layer",
        new LayerConfig("vertical_layer", IZHIKEVICH, "bursting"),
        new ConnectionConfig(true, 0, 10, DIVERGENT, ADD,
            new RandomWeightConfig(0.5),
            new ArborizedConfig(vertical_spread,horizontal_spread,1,1)));
    // Vertical cluster inhibitory layer
    structure->connect_layers_expected("convergent_layer",
        new LayerConfig("vertical_inhibitory", IZHIKEVICH, "fast"),
        new ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(0.5),
            new ArborizedConfig(vertical_spread,horizontal_spread,1,1)));
    structure->connect_layers("vertical_inhibitory", "vertical_layer",
        new ConnectionConfig(false, 0, 5, DIVERGENT, DIV,
            new RandomWeightConfig(5),
            new ArborizedConfig(vertical_spread,horizontal_spread,1,1)));

    // Block cluster layer
    structure->connect_layers_expected("vertical_layer",
        new LayerConfig("cluster_layer", IZHIKEVICH, "bursting", 0.5),
        new ConnectionConfig(true, 0, 10, DIVERGENT, ADD,
            new RandomWeightConfig(0.5),
            new ArborizedConfig(vertical_spread,horizontal_spread,1,1)));
    // Block cluster inhibitory layer
    structure->connect_layers_expected("vertical_layer",
        new LayerConfig("cluster_inhibitory", IZHIKEVICH, "fast"),
        new ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(0.5),
            new ArborizedConfig(vertical_spread,horizontal_spread,1,1)));
    structure->connect_layers("cluster_inhibitory", "cluster_layer",
        new ConnectionConfig(false, 0, 5, DIVERGENT, DIV,
            new RandomWeightConfig(5),
            new ArborizedConfig(vertical_spread,horizontal_spread,1,1)));

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
        new LayerConfig("motion_up", IZHIKEVICH, "bursting", 1),
        new ConnectionConfig(false, 5, 10, CONVERGENT, ADD,
            new RandomWeightConfig(pos_strength),
            new ArborizedConfig(1,mot_f_size,1,mot_stride,offset,offset)));
    structure->connect_layers("cluster_layer", "motion_up",
        new ConnectionConfig(false, 5, 10, CONVERGENT, DIV,
            new RandomWeightConfig(neg_strength),
            new ArborizedConfig(1,mot_f_size,1,mot_stride,-offset,-offset)));
    structure->connect_layers("cluster_layer", "motion_up",
        new ConnectionConfig(false, 0, 10, CONVERGENT, MULT,
            new RandomWeightConfig(mid_strength),
            new ArborizedConfig(1,mot_f_size,1,mot_stride)));
    structure->connect_layers_expected("motion_up",
        new LayerConfig("motion_up_pool", IZHIKEVICH, "chattering", 1),
        new ConnectionConfig(false, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(pool_strength),
            new ArborizedConfig(1,pool_f_size,1,pool_stride)));

    structure->connect_layers_expected("cluster_layer",
        new LayerConfig("motion_down", IZHIKEVICH, "bursting", 1),
        new ConnectionConfig(false, 5, 10, CONVERGENT, ADD,
            new RandomWeightConfig(pos_strength),
            new ArborizedConfig(1,mot_f_size,1,mot_stride,-offset,-offset)));
    structure->connect_layers("cluster_layer", "motion_down",
        new ConnectionConfig(false, 5, 10, CONVERGENT, DIV,
            new RandomWeightConfig(neg_strength),
            new ArborizedConfig(1,mot_f_size,1,mot_stride,offset,offset)));
    structure->connect_layers("cluster_layer", "motion_down",
        new ConnectionConfig(false, 0, 10, CONVERGENT, MULT,
            new RandomWeightConfig(mid_strength),
            new ArborizedConfig(1,mot_f_size,1,mot_stride)));
    structure->connect_layers_expected("motion_down",
        new LayerConfig("motion_down_pool", IZHIKEVICH, "chattering", 1),
        new ConnectionConfig(false, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(pool_strength),
            new ArborizedConfig(1,pool_f_size,1,pool_stride)));
    */

    // Modules
    std::string output_name = "visualizer_output";

    //structure->add_module("input_layer", "random_input", "10 500");
    //structure->add_module("input_layer", "csv_input",
    //    "./resources/substitute.csv 0 1 0.1");
    //structure->add_module("input_layer", "csv_input",
    //    "./resources/sample.csv 0 1 0.01");
    structure->add_module("input_layer", "csv_input",
        "./resources/substitute.csv 0 1 0.01");

    structure->add_module("input_layer", output_name);
    structure->add_module("convergent_layer", output_name);
    structure->add_module("convergent_layer_inhibitory", output_name);

    structure->add_module("convergent_layer", "csv_output");

    /*
    structure->add_module("vertical_layer", output_name);
    structure->add_module("cluster_layer", output_name);
    structure->add_module("motion_up", output_name);
    structure->add_module("motion_up_pool", output_name);
    structure->add_module("motion_down", output_name);
    structure->add_module("motion_down_pool", output_name);
    */

    std::cout << "Speech test......\n";
    print_model(model);
    Clock clock((float)240.0);
    //Clock clock(true);
    clock.run(model, 1000000, true);
    std::cout << "\n";

    delete model;
}

void re_speech_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("speech", SEQUENTIAL);

    // Input layer
    structure->add_layer(new LayerConfig("input_layer", RELAY, 1, 41));

    // Convergent layers
    structure->connect_layers_matching("input_layer",
        new LayerConfig("convergent_layer", HEBBIAN_RATE_ENCODING),
        new ConnectionConfig(false, 2, 100, CONVERGENT, ADD,
            new FlatWeightConfig(35),
            new ArborizedConfig(1,1,1,1)));

    structure->connect_layers_matching("input_layer",
        new LayerConfig("convergent_layer_inhibitory", HEBBIAN_RATE_ENCODING),
        new ConnectionConfig(false, 0, 100, CONVERGENT, ADD,
            new FlatWeightConfig(3),
            new ArborizedConfig(1,10,1,1)));
    structure->connect_layers("convergent_layer_inhibitory",
        "convergent_layer",
        new ConnectionConfig(false, 0, 100, ONE_TO_ONE, SUB,
            new FlatWeightConfig(1)));

    // Modules
    std::string output_name = "visualizer_output";

    //structure->add_module("input_layer", "random_input", "10 500");
    //structure->add_module("input_layer", "csv_input",
    //    "./resources/substitute.csv 0 1 1");
    //structure->add_module("input_layer", "csv_input",
    //    "./resources/sample.csv 0 1 1");
    structure->add_module("input_layer", "csv_input",
        "./resources/speech.csv 0 1 1");

    structure->add_module("input_layer", output_name);
    structure->add_module("convergent_layer", output_name);
    structure->add_module("convergent_layer_inhibitory", output_name);

    structure->add_module("convergent_layer", "csv_output");

    std::cout << "Rate encoding speech test......\n";
    print_model(model);
    Clock clock((float)240.0);
    //Clock clock(true);
    clock.run(model, 1000000, true);
    std::cout << "\n";

    delete model;
}

void second_order_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("second_order");

    // Size of large receptive field
    int field_size = 100;

    const char* image_path = "resources/bird-head.jpg";
    structure->add_layer_from_image(image_path,
        new LayerConfig("image", IZHIKEVICH, "default"));

    structure->connect_layers_expected("image",
        new LayerConfig("pool", IZHIKEVICH, "default"),
        new ConnectionConfig(false, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(1),
            new ArborizedConfig(10,3)));

    structure->connect_layers_matching("pool",
        new LayerConfig("out", IZHIKEVICH, "default"),
        new ConnectionConfig(false, 0, 100, CONVERGENT, ADD,
            new FlatWeightConfig(100),
            new ArborizedConfig(field_size,1)));

    structure->get_dendritic_root("out")->set_second_order();

    structure->add_layer(new LayerConfig("predict", IZHIKEVICH, field_size, field_size, "default"));
    structure->connect_layers("predict", "out",
        new ConnectionConfig(false, 0, 100, CONVERGENT, MULT,
            new FlatWeightConfig(1),
            new ArborizedConfig(field_size,0)));

    // Modules
    std::string output_name = "visualizer_output";
    //std::string output_name = "dummy_output";

    structure->add_module("image", "image_input", image_path);
    structure->add_module("predict", "one_hot_random_input", "100 50");
    structure->add_module("image", output_name);
    structure->add_module("pool", output_name);
    structure->add_module("predict", output_name);
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
        working_memory_test();
        //dendritic_test();
        //hh_test();
        //cc_test();
        //re_test();
        //mnist_test();
        //divergent_test();
        //speech_test();
        //re_speech_test();
        //second_order_test();

        return 0;
    } catch (const char* msg) {
        printf("\n\nEXPECTED: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
