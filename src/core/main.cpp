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
#include "maze_game.h"

#define IZHIKEVICH "izhikevich"
#define HODGKIN_HUXLEY "hodgkin_huxley"
#define HEBBIAN_RATE_ENCODING "hebbian_rate_encoding"
#define BACKPROP_RATE_ENCODING "backprop_rate_encoding"
#define RELAY "relay"

#define IZ_INIT "init"

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

Model* build_working_memory_model(std::string neural_model) {
    /* Construct the model */
    Model *model = new Model();
    Structure *main_structure = model->add_structure("working memory", PARALLEL);

    int thal_ratio = 1;
    int cortex_size = 128 + 64;
    int thal_size = cortex_size / thal_ratio;

    float inter_conn_ratio = 0.2;
    float ff_conn_ratio = 0.5;
    float gamma_conn_ratio = 1.0;

    float ff_noise = 0.0;
    float thal_noise = 0.0;
    float cortex_noise = 1.0;

    bool exc_plastic = false;
    bool inh_plastic = false;
    int exc_delay = 3;
    int inh_delay = 0;

    int sensory_center = 15;
    int sensory_surround = 15;
    int inter_cortex_center = 15;
    int inter_cortex_surround = 25;
    int gamma_center = 3;
    int gamma_surround = 3;

    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    // Feedforward circuit
    main_structure->add_layer(new LayerConfig("input_layer", RELAY, 1, 10));
    main_structure->add_layer((new LayerConfig("feedforward",
        neural_model, cortex_size, cortex_size, ff_noise))
            ->set_property(IZ_INIT, "regular"));

    // Thalamic relay
    main_structure->add_layer((new LayerConfig("tl1_thalamus",
        neural_model, 1, 1, thal_noise))
            ->set_property(IZ_INIT, "thalamo_cortical"));

    // Feedforward input
    main_structure->connect_layers("input_layer", "feedforward",
        new ConnectionConfig(false, exc_delay, 1, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(100, 0.1)));

    std::vector<Structure*> sub_structures;
    for (int i = 0 ; i < 2 ; ++i) {
        Structure *sub_structure =
            model->add_structure(std::to_string(i), PARALLEL);
        sub_structures.push_back(sub_structure);

        // Thalamic gamma nucleus
        sub_structure->add_layer((new LayerConfig("gamma_thalamus",
            neural_model, thal_size, thal_size, thal_noise))
                ->set_property(IZ_INIT, "thalamo_cortical"));

        // Cortical layers
        sub_structure->add_layer((new LayerConfig("3_cortex",
            neural_model, cortex_size, cortex_size, cortex_noise))
                ->set_property(IZ_INIT, "random positive"));
        sub_structure->add_layer((new LayerConfig("6_cortex",
            neural_model, cortex_size, cortex_size, cortex_noise))
                ->set_property(IZ_INIT, "random positive"));

        // Cortico-cortical connectivity
        sub_structure->connect_layers("3_cortex", "6_cortex",
            new ConnectionConfig(exc_plastic, exc_delay, 4, CONVERGENT, ADD,
                new FlatWeightConfig(1, inter_conn_ratio),
                new ArborizedConfig(inter_cortex_center,1)));

        sub_structure->connect_layers("6_cortex", "3_cortex",
            new ConnectionConfig(exc_plastic, exc_delay, 4, CONVERGENT, ADD,
                new FlatWeightConfig(1, inter_conn_ratio),
                new ArborizedConfig(inter_cortex_center,1)));

        sub_structure->connect_layers("3_cortex", "6_cortex",
            new ConnectionConfig(inh_plastic, inh_delay, 4, CONVERGENT, SUB,
                new SurroundWeightConfig(inter_cortex_center,
                    new FlatWeightConfig(1, inter_conn_ratio)),
                new ArborizedConfig(inter_cortex_surround,1)));
        sub_structure->connect_layers("6_cortex", "3_cortex",
            new ConnectionConfig(inh_plastic, inh_delay, 4, CONVERGENT, SUB,
                new SurroundWeightConfig(inter_cortex_center,
                    new FlatWeightConfig(1, inter_conn_ratio)),
                new ArborizedConfig(inter_cortex_surround,1)));

        // Gamma connectivity
        sub_structure->connect_layers("gamma_thalamus", "3_cortex",
            new ConnectionConfig(exc_plastic, 10 + exc_delay, 4, CONVERGENT, ADD,
                new FlatWeightConfig(1*thal_ratio, gamma_conn_ratio),
                new ArborizedConfig(gamma_center,1)));
        sub_structure->connect_layers("gamma_thalamus", "3_cortex",
            new ConnectionConfig(inh_plastic, 10 + inh_delay, 4, CONVERGENT, SUB,
                new SurroundWeightConfig(gamma_center,
                    new FlatWeightConfig(1*thal_ratio, gamma_conn_ratio)),
                new ArborizedConfig(gamma_surround,1)));
        sub_structure->connect_layers("6_cortex", "gamma_thalamus",
            new ConnectionConfig(exc_plastic, 10 + exc_delay, 4, CONVERGENT, ADD,
                new FlatWeightConfig(1*thal_ratio, gamma_conn_ratio),
                new ArborizedConfig(gamma_center,1)));
        sub_structure->connect_layers("6_cortex", "gamma_thalamus",
            new ConnectionConfig(inh_plastic, 10 + inh_delay, 4, CONVERGENT, SUB,
                new SurroundWeightConfig(gamma_center,
                    new FlatWeightConfig(1*thal_ratio, gamma_conn_ratio)),
                new ArborizedConfig(gamma_surround,1)));

        // Thalamocortical control connectivity
        Structure::connect(main_structure, "tl1_thalamus",
            sub_structure, "3_cortex",
            new ConnectionConfig(false, 0, 4, FULLY_CONNECTED, MULT,
                new FlatWeightConfig(1*thal_ratio, inter_conn_ratio)));
        Structure::connect(main_structure, "tl1_thalamus",
            sub_structure, "6_cortex",
            new ConnectionConfig(false, 0, 4, FULLY_CONNECTED, MULT,
                new FlatWeightConfig(1*thal_ratio, inter_conn_ratio)));

        sub_structure->add_module("3_cortex", output_name, "8");
        sub_structure->add_module("6_cortex", output_name, "8");
        sub_structure->add_module("gamma_thalamus", output_name, "8");
    }

    // Sensory relay to cortex
    for (int i = 0 ; i < 1 ; ++i) {
        /*
        Structure::connect(main_structure, "feedforward",
            sub_structures[i], "3_cortex",
            new ConnectionConfig(exc_plastic, 0, 4, CONVERGENT, ADD,
                new RandomWeightConfig(1*thal_ratio, ff_conn_ratio),
                new ArborizedConfig(sensory_center,1)));
        Structure::connect(main_structure, "feedforward",
            sub_structures[i], "3_cortex",
            new ConnectionConfig(inh_plastic, 0, 4, CONVERGENT, SUB,
                new SurroundWeightConfig(sensory_center,
                    new RandomWeightConfig(1*thal_ratio, ff_conn_ratio)),
                new ArborizedConfig(sensory_surround,1)));
        */
        Structure::connect(main_structure, "feedforward",
            sub_structures[i], "3_cortex",
            new ConnectionConfig(exc_plastic, 0, 4, ONE_TO_ONE, ADD,
                new FlatWeightConfig(1*thal_ratio, ff_conn_ratio)));
    }

    Structure::connect(sub_structures[0], "3_cortex",
        sub_structures[1], "3_cortex",
        new ConnectionConfig(exc_plastic, 0, 4, CONVERGENT, ADD,
            new FlatWeightConfig(1*thal_ratio, ff_conn_ratio),
            new ArborizedConfig(sensory_center,1)));
    Structure::connect(sub_structures[0], "3_cortex",
        sub_structures[1], "3_cortex",
        new ConnectionConfig(inh_plastic, 0, 4, CONVERGENT, SUB,
            new SurroundWeightConfig(sensory_center,
                new FlatWeightConfig(1*thal_ratio, ff_conn_ratio)),
            new ArborizedConfig(sensory_surround,1)));

    // Modules
    main_structure->add_module("input_layer", "one_hot_random_input", "1 5000");
    main_structure->add_module("tl1_thalamus", "random_input", "1 500");
    main_structure->add_module("feedforward", output_name, "8");

    return model;
}

void working_memory_test() {
    Model *model;

    std::cout << "Working memory...\n";
    model = build_working_memory_model(IZHIKEVICH);
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
    structure->add_layer((new LayerConfig("input_layer",
        IZHIKEVICH, 28, 28))
            ->set_property(IZ_INIT, "default"));

    int num_hidden = 10;
    for (int i = 0; i < num_hidden; ++i) {
        structure->add_layer((new LayerConfig(std::to_string(i),
            IZHIKEVICH, 28, 28, 0.5))
                ->set_property(IZ_INIT, "default"));
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

void speech_test() {
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
        (new LayerConfig("image", IZHIKEVICH))
            ->set_property(IZ_INIT, "default"));

    structure->connect_layers_expected("image",
        (new LayerConfig("pool", IZHIKEVICH))
            ->set_property(IZ_INIT, "default"),
        new ConnectionConfig(false, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(1),
            new ArborizedConfig(10,3)));

    structure->connect_layers_matching("pool",
        (new LayerConfig("out", IZHIKEVICH))
            ->set_property(IZ_INIT, "default"),
        new ConnectionConfig(false, 0, 100, CONVERGENT, ADD,
            new FlatWeightConfig(100),
            new ArborizedConfig(field_size,1)));

    structure->get_dendritic_root("out")->set_second_order();

    structure->add_layer((new LayerConfig("predict",
        IZHIKEVICH, field_size, field_size))
            ->set_property(IZ_INIT, "default"));
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

void maze_game_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("maze_game");

    int board_size = MazeGame::get_instance(true)->get_board_dim();

    structure->add_layer(new LayerConfig("player_layer",
        RELAY, board_size, board_size));
    structure->add_layer(new LayerConfig("goal_layer",
        RELAY, board_size, board_size));

    structure->add_layer(new LayerConfig("movement_layer", RELAY, 1, 4));

    // Modules
    std::string output_name = "visualizer_output";
    //std::string output_name = "print_output";
    //std::string output_name = "dummy_output";

    structure->add_module("player_layer", "maze_input", "player");
    structure->add_module("goal_layer", "maze_input", "goal");
    structure->add_module("movement_layer", "one_hot_random_input", "1 1");
    structure->add_module("movement_layer", "maze_output");

    structure->add_module("player_layer", output_name, "8");
    structure->add_module("goal_layer", output_name, "8");
    structure->add_module("movement_layer", output_name, "8");

    std::cout << "Maze game test......\n";
    print_model(model);
    Clock clock((float)3.0);
    //Clock clock(true);
    clock.run(model, 1000000, true);
    std::cout << "\n";

    delete model;
}

void symbol_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = model->add_structure("symbol");

    int cortex_size = 64;
    float noise = 1.0;
    bool exc_plastic = true;
    bool inh_plastic = false;
    std::string output_name = "visualizer_output";

    structure->add_layer(new LayerConfig("input_layer", RELAY, 1, 10));

    // Layers
    for (int i = 0 ; i < 2 ; ++i) {
        std::string pos_name = std::string("pos") + std::to_string(i);
        std::string neg_name = std::string("neg") + std::to_string(i);
        structure->add_layer((new LayerConfig(pos_name,
            IZHIKEVICH, cortex_size, cortex_size, noise))
                ->set_property(IZ_INIT, "regular"));
        structure->add_layer((new LayerConfig(neg_name,
            IZHIKEVICH, cortex_size/2, cortex_size/2, noise))
                ->set_property(IZ_INIT, "fast"));

        structure->connect_layers(pos_name, pos_name,
            new ConnectionConfig(exc_plastic, 0, 4, FULLY_CONNECTED, ADD,
                new GaussianWeightConfig(1, 0.3, 0.1)));
        structure->connect_layers(pos_name, neg_name,
            new ConnectionConfig(false, 0, 4*4, FULLY_CONNECTED, ADD,
                new GaussianWeightConfig(4*1, 4*0.3, 0.1)));

        structure->connect_layers(neg_name, pos_name,
            new ConnectionConfig(inh_plastic, 0, 4*4, FULLY_CONNECTED, SUB,
                new GaussianWeightConfig(4*1, 4*0.3, 0.1)));
        structure->add_module(pos_name, output_name, "8");
        structure->add_module(neg_name, output_name, "8");
    }

    // Chain inputs
    structure->connect_layers("input_layer", "pos0",
        new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(100, 0.01)));
    structure->connect_layers("pos0", "pos1",
        new ConnectionConfig(exc_plastic, 0, 4*10, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(1, 0.3, 0.01)));


    // Modules
    structure->add_module("input_layer", "one_hot_random_input", "1 500");

    std::cout << "Symbol test......\n";
    print_model(model);
    //Clock clock((float)120.0);
    Clock clock(true);
    clock.run(model, 1000000, true);
    std::cout << "\n";

    delete model;
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(nullptr));

    try {
        //working_memory_test();
        //mnist_test();
        //speech_test();
        //second_order_test();
        //maze_game_test();
        symbol_test();

        return 0;
    } catch (const char* msg) {
        printf("\n\nEXPECTED: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
