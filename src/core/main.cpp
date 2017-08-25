#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model/auditory_cortex.h"
#include "model/maze_cortex.h"
#include "model/model.h"
#include "model/model_builder.h"
#include "model/column.h"
#include "model/weight_config.h"
#include "io/module/module.h"
#include "state/state.h"
#include "util/tools.h"
#include "clock.h"
#include "maze_game.h"

#define IZHIKEVICH "izhikevich"
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
            printf("%-20s   ", (layer->structure->name + "->" + layer->name).c_str());
            std::cout << ((layer->is_input()) ? "I " : "  ");
            std::cout << ((layer->is_output()) ? "O " : "  ");
            std::cout << ((layer->is_expected()) ? "E " : "  ");
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void run_simulation(Model *model, int iterations, bool verbose) {
    // Calculate ideal refresh rate, run for iterations
    Clock clock(true);
    delete clock.run(model, iterations, verbose);

    // Benchmark the network
    // Use max refresh rate possible
    // Run for 100 iterations
    //Clock clock(false);  // No refresh rate synchronization
    //delete clock.run(model, 100, verbose);
}

void old_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = new Structure("old");
    model->add_structure(structure);

    std::string model_name = "leaky_izhikevich";

    int resolution = 128;
    // structure->add_layer((new LayerConfig(
    //     "input_layer", model_name, 1, 10))
    //        ->set_property(IZ_INIT, "default"));
    structure->add_layer(
        (new LayerConfig(
            "exc_thalamus", model_name, resolution, resolution,
            (new NoiseConfig("poisson"))))
        ->set_property(IZ_INIT, "thalamo_cortical"));
    // structure->add_layer((new LayerConfig(
    //     "inh_thalamus", model_name, resolution/2, resolution/2))
    //         ->set_property("spacing", "0.2")
    //         ->set_property(IZ_INIT, "random negative"));
    structure->add_layer((new LayerConfig(
        "exc_cortex", model_name, resolution, resolution))
            ->set_property(IZ_INIT, "random positive"));
    structure->add_layer((new LayerConfig(
        "inh_cortex", model_name, resolution/2, resolution/2))
            ->set_property("spacing", "0.2")
            ->set_property(IZ_INIT, "random negative"));

    /* Forward excitatory pathway */
    // structure->connect_layers("input_layer", "exc_thalamus",
    //     (new ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
    //         new RandomWeightConfig(1, 0.01)))
    //     ->set_property("myelinated", "true"));
    structure->connect_layers("exc_thalamus", "exc_cortex",
        (new ConnectionConfig(true, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_property("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(15,1,true)));
    structure->connect_layers("exc_cortex", "exc_cortex",
        (new ConnectionConfig(true, 2, 5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_property("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(31,1,true)));

    /* Cortical inhibitory loop */
    structure->connect_layers("exc_cortex", "inh_cortex",
        (new ConnectionConfig(false, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_property("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(31,2,true)));
    structure->connect_layers("inh_cortex", "exc_cortex",
        (new ConnectionConfig(false, 0, 1, DIVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_property("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(5,2,true)));

    /* Cortico-thalamic inhibitory loop */
    /*
    structure->connect_layers("exc_cortex", "inh_thalamus",
        (new ConnectionConfig(true, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_property("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(7,2,true)));
    structure->connect_layers("inh_thalamus", "exc_thalamus",
        (new ConnectionConfig(false, 0, 1, DIVERGENT, SUB,
            new FlatWeightConfig(0.5, 0.1)))
        ->set_property("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(5,2,true)));
    */


    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    // structure->add_module("input_layer",
    //     (new ModuleConfig("random_input"))
    //     ->set_property("max", "5")
    //     ->set_property("rate", "1000000")
    //     ->set_property("verbose", "true"));
    structure->add_module("exc_thalamus",
        new ModuleConfig(output_name));
    structure->add_module("exc_cortex",
        new ModuleConfig(output_name));
    structure->add_module("exc_thalamus",
        new ModuleConfig("heatmap"));
    structure->add_module("exc_cortex",
        new ModuleConfig("heatmap"));

    // structure->add_module("inh_thalamus",
    //     new ModuleConfig(output_name));
    structure->add_module("inh_cortex",
        new ModuleConfig(output_name));
    // structure->add_module("inh_thalamus",
    //     new ModuleConfig("heatmap"));
    structure->add_module("inh_cortex",
        new ModuleConfig("heatmap"));

    print_model(model);
    Clock clock(true);
    delete clock.run(model, 1000000, true);
    delete model;
}

void simple_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = new Structure("simple");
    model->add_structure(structure);

    std::string model_name = "leaky_izhikevich";

    int exc_field = 25;
    int inh_field = 15;

    int resolution = 96;
    structure->add_layer((new LayerConfig(
        "input_layer", model_name, 1, 10))
            ->set_property(IZ_INIT, "regular"));
    structure->add_layer(
        (new LayerConfig(
            "hid_1", model_name, resolution, resolution,
            (new NoiseConfig("poisson"))
                ->set_property("value", "20")
                ->set_property("rate", "1")))
            ->set_property(IZ_INIT, "regular"));
    structure->add_layer((new LayerConfig(
        "hid_2", model_name, resolution, resolution))
            ->set_property(IZ_INIT, "regular"));

    /* Forward excitatory pathway */
    /*
    structure->connect_layers("input_layer", "hid_1",
        (new ConnectionConfig(false, 0, 0.5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(1, 0.05)))
        ->set_property("myelinated", "true"));
    */

    structure->connect_layers("hid_1", "hid_2",
        (new ConnectionConfig(true, 10, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field,1,true)));
    /*
    structure->connect_layers("hid_1", "hid_2",
        (new ConnectionConfig(false, 10, 0.5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field,1,true)));
    */

    /* Recurrent self connectivity */
    structure->connect_layers("hid_1", "hid_1",
        (new ConnectionConfig(true, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field,1,true)));
    structure->connect_layers("hid_1", "hid_1",
        (new ConnectionConfig(false, 0, 0.5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field,1,true)));

    structure->connect_layers("hid_2", "hid_2",
        (new ConnectionConfig(true, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field,1,true)));
    structure->connect_layers("hid_2", "hid_2",
        (new ConnectionConfig(false, 0, 0.5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field,1,true)));

    /* Feedback connectivity */
    structure->connect_layers("hid_2", "hid_1",
        (new ConnectionConfig(true, 10, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field,1,true)));
    /*
    structure->connect_layers("hid_2", "hid_1",
        (new ConnectionConfig(false, 10, 0.5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field,1,true)));
    */

    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    structure->add_module("input_layer",
        (new ModuleConfig("one_hot_random_input"))
            ->set_property("max", "4")
            ->set_property("rate", "1000000"));

    structure->add_module("input_layer",
        new ModuleConfig(output_name));
    structure->add_module("hid_1",
        new ModuleConfig(output_name));
    structure->add_module("hid_2",
        new ModuleConfig(output_name));

    structure->add_module("input_layer",
        new ModuleConfig("heatmap"));
    structure->add_module("hid_1",
        new ModuleConfig("heatmap"));
    structure->add_module("hid_2",
        new ModuleConfig("heatmap"));


    Clock clock(true);
    std::string filename = "simple.bin";

    if (State::exists(filename)) {
        print_model(model);
        auto state = new State(model);
        state->load("simple.bin");
        state = clock.run(model, 1000000, true, state);
        delete state;
    } else {
        print_model(model);
        auto state = new State(model);
        state = clock.run(model, 500000, true, state, true);
        state->save("simple.bin");
        delete state;
    }
}

void single_field_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = new Structure("single field");
    model->add_structure(structure);

    std::string model_name = "leaky_izhikevich";

    structure->add_layer((new LayerConfig(
        "exc_field", model_name, 40, 40))
            ->set_property(IZ_INIT, "random positive"));
    structure->add_layer((new LayerConfig(
        "inh_field", model_name, 10, 40))
            ->set_property(IZ_INIT, "random negative"));

    structure->connect_layers("exc_field", "exc_field",
        (new ConnectionConfig(true, 0, 10, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(.6, 0.1)))
        ->set_property("random delay", "20"));
    structure->connect_layers("exc_field", "inh_field",
        (new ConnectionConfig(true, 0, 10, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(.6, 0.1)))
        ->set_property("random delay", "20"));

    structure->connect_layers("inh_field", "inh_field",
        (new ConnectionConfig(false, 0, 10, FULLY_CONNECTED, SUB,
            new FlatWeightConfig(.5, 0.1)))
        ->set_property("myelinated", "true"));
    structure->connect_layers("inh_field", "exc_field",
        (new ConnectionConfig(false, 0, 10, FULLY_CONNECTED, SUB,
            new FlatWeightConfig(.5, 0.1)))
        ->set_property("myelinated", "true"));

    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    structure->add_module("exc_field",
        (new ModuleConfig("one_hot_random_input"))
            ->set_property("max", "20")
            ->set_property("rate", "1")
            ->set_property("verbose", "false"));
    structure->add_module("exc_field",
        new ModuleConfig(output_name));
    structure->add_module("inh_field",
        new ModuleConfig(output_name));

    structure->add_module("exc_field",
        new ModuleConfig("heatmap"));
    structure->add_module("inh_field",
        new ModuleConfig("heatmap"));

    print_model(model);
    Clock clock(true);
    delete clock.run(model, 1000000, true);
    delete model;
}

void mnist_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = new Structure("mnist");
    model->add_structure(structure);

    int resolution = 1024;
    structure->add_layer((new LayerConfig("input_layer",
        IZHIKEVICH, 28, 28))
            ->set_property(IZ_INIT, "regular"));

    int num_hidden = 10;
    for (int i = 0; i < num_hidden; ++i) {
        structure->add_layer(
            (new LayerConfig(std::to_string(i),
                "leaky_izhikevich", 28, 28,
                (new NoiseConfig("normal"))
                    ->set_property("mean", "0.5")
                    ->set_property("std_dev", "0.1")))
                ->set_property(IZ_INIT, "regular"));
        structure->connect_layers("input_layer", std::to_string(i),
            (new ConnectionConfig(true, 0, 0.5, FULLY_CONNECTED, ADD,
                new FlatWeightConfig(0.1, 0.1)))
            ->set_arborized_config(new ArborizedConfig(9,1,true)));

        structure->connect_layers(std::to_string(i), std::to_string(i),
            (new ConnectionConfig(true, 0, 0.5, FULLY_CONNECTED, ADD,
                new RandomWeightConfig(0.1, 0.1)))
            ->set_arborized_config(new ArborizedConfig(9,1,true)));
        structure->connect_layers(std::to_string(i), std::to_string(i),
            (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, SUB,
                new FlatWeightConfig(0.5, 0.1)))
            ->set_arborized_config(new ArborizedConfig(11,1,true)));
    }

    for (int i = 0; i < num_hidden; ++i)
        for (int j = 0; j < num_hidden; ++j)
            if (i != j)
                structure->connect_layers(std::to_string(i), std::to_string(j),
                    new ConnectionConfig(false, 0, 1, ONE_TO_ONE, SUB,
                        new FlatWeightConfig(0.5)));

    // Modules
    structure->add_module("input_layer",
        new ModuleConfig("csv_input", "/HDD/datasets/mnist/mnist_test.csv 1 5000 25"));
    structure->add_module("input_layer", new ModuleConfig("visualizer_output"));
    structure->add_module("input_layer", new ModuleConfig("heatmap"));
    for (int i = 0; i < num_hidden; ++i) {
        structure->add_module(std::to_string(i), new ModuleConfig("visualizer_output"));
        structure->add_module(std::to_string(i), new ModuleConfig("heatmap"));
    }

    std::cout << "CSV test......\n";
    print_model(model);
    run_simulation(model, 100000, true);
    std::cout << "\n";

    delete model;
}

void speech_train() {
    /* Construct the model */
    Model *model = new Model();

    AuditoryCortex *auditory_cortex = new AuditoryCortex(model, 41, 7);
    auditory_cortex->add_input("3b_pos", "speech_input",
        new ModuleConfig("csv_input", "./resources/hearing.csv 0 1 0.25"));
    //auditory_cortex->add_input("3b_pos", "speech_input",
    //    new ModuleConfig("csv_input", "./resources/substitute.csv 0 1 0.25"));
    //auditory_cortex->add_module_all(new ModuleConfig("visualizer_output"));
    //auditory_cortex->add_module_all(new ModuleConfig("heatmap"));

    // Modules
    //structure->add_module("convergent_layer", new ModuleConfig("csv_output"));

    std::cout << "Speech train......\n";
    print_model(model);
    Clock clock(false);

    auto state = clock.run(model, 1635324, true);
    std::cout << "\n";
    state->save("hearing-spread-dual.bin");

    delete state;
    delete model;
}

void speech_test(std::string filename) {
    /* Construct the model */
    Model *model = new Model();

    AuditoryCortex *auditory_cortex = new AuditoryCortex(model, 41, 7);
    auditory_cortex->add_input("3b_pos", "speech_input",
        new ModuleConfig("csv_input", filename + " 0 1 0.5"));
    //auditory_cortex->add_module_all(new ModuleConfig("visualizer_output"));
    //auditory_cortex->add_module_all(new ModuleConfig("heatmap"));
    auditory_cortex->add_module("5a_pos", new ModuleConfig("csv_output"));

    std::cout << "Speech test......\n";
    print_model(model);
    Clock clock(true);

    auto state = new State(model);
    state->load("hearing-spread.bin");
    state = clock.run(model, 717, false, state);

    delete state;
    std::cout << "\n";

    delete model;
}

void maze_game_test() {
    /* Construct the model */
    Model *model = new Model();
    int board_dim = 5;
    MazeGame::get_instance(true)->set_board_dim(board_dim);

    MazeCortex *maze_cortex = new MazeCortex(model, board_dim, 16);
    maze_cortex->add_module_all(new ModuleConfig("visualizer_output"));
    maze_cortex->add_module_all(new ModuleConfig("heatmap"));

    std::cout << "Maze game test......\n";
    print_model(model);
    Clock clock(true);
    //Clock clock(10.0f);

    MazeGame::get_instance(true)->set_board_dim(board_dim);
    delete clock.run(model, 1000000, true);

    //delete clock.run(model, 100, true);
    std::cout << "\n";

    delete model;
}

void game_of_life_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = new Structure("game_of_life");
    model->add_structure(structure);
    std::string model_name = "game_of_life";

    // Contour rules
    // R4,C0,M1,S41..81, B41..81,NM
    // R7,C0,M1,S113..225, B113..225,NM

    // Game parameters
    bool wrap = true;
    int board_dim = 1024;
    int neighborhood_size = 5; 15;
    int survival_min = 2; 113;
    int survival_max = 3; 225;
    int birth_min = 3; 113;
    int birth_max = 4; 225;

    // Input parameters
    bool one_step = false;
    float one_step_fraction = 0.5;
    float random_fraction = 0.5;
    float rate = 1000;

    structure->add_layer((new LayerConfig(
        "board", model_name, board_dim, board_dim,
        (new NoiseConfig("poisson"))
        ->set_property("value", std::to_string(birth_min))
        ->set_property("rate", "0.5")))
            ->set_property("survival_min", std::to_string(survival_min))
            ->set_property("survival_max", std::to_string(survival_max))
            ->set_property("birth_min", std::to_string(birth_min))
            ->set_property("birth_max", std::to_string(birth_max)));
    structure->connect_layers("board", "board",
        (new ConnectionConfig(false, 0, 1.0, CONVOLUTIONAL, ADD,
            new SurroundWeightConfig(1,1, new FlatWeightConfig(1))))
        ->set_arborized_config(
            new ArborizedConfig(neighborhood_size, 1, -neighborhood_size/2, wrap)));
    structure->connect_layers("board", "board",
        (new ConnectionConfig(false, 0, 1.0, CONVOLUTIONAL, SUB,
            new SurroundWeightConfig(neighborhood_size,neighborhood_size,
                new FlatWeightConfig(1))))
        ->set_arborized_config(
            new ArborizedConfig(neighborhood_size+2, 1, (-neighborhood_size+1)/2, wrap)));

    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    /*
    if (one_step)
        // Single Initial State
        structure->add_module("board",
            (new ModuleConfig("one_step_input"))
                ->set_property("max", std::to_string(birth_min))
                ->set_property("uniform", "true")
                ->set_property("verbose", "false")
                ->set_property("fraction", std::to_string(one_step_fraction)));
    else
        // Refresh state
        structure->add_module("board",
            (new ModuleConfig("random_input"))
                ->set_property("max", std::to_string(birth_min))
                ->set_property("rate", std::to_string(rate))
                ->set_property("uniform", "true")
                ->set_property("clear", "true")
                ->set_property("verbose", "false")
                ->set_property("fraction", std::to_string(random_fraction)));
    */

    structure->add_module("board",
        new ModuleConfig(output_name));

    Clock clock(true);
    //Clock clock(1.0f);
    print_model(model);
    delete clock.run(model, 500000, true);
}

void working_memory_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *main_structure = new Structure("working memory", PARALLEL);
    model->add_structure(main_structure);

    bool wrap = true;
    int num_cortical_regions = 3;

    int thal_ratio = 1;
    int cortex_size = 128;
    int thal_size = cortex_size / thal_ratio;

    float cortex_noise = 1.0;
    float cortex_noise_stdev = 0.3;

    bool exc_plastic = true;
    int exc_delay = 0;
    int inh_delay = 3;

    int ff_center = 15;
    int ff_surround = 19;
    int intra_cortex_center = 15;
    int intra_cortex_surround = 19;
    int gamma_center = 9;
    int gamma_surround = 15;

    float fraction = 0.1;

    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    // Feedforward circuit
    main_structure->add_layer(
        (new LayerConfig("feedforward",
            IZHIKEVICH, cortex_size, cortex_size,
            new NoiseConfig("poisson")))
        ->set_property(IZ_INIT, "regular"));

    // Thalamic relay
    main_structure->add_layer(
        (new LayerConfig("tl1_thalamus",
            IZHIKEVICH, 1, 1))
        ->set_property(IZ_INIT, "thalamo_cortical"));

    std::vector<Structure*> sub_structures;
    for (int i = 0 ; i < num_cortical_regions ; ++i) {
        Structure *sub_structure = new Structure(std::to_string(i), PARALLEL);

        // Thalamic gamma nucleus
        /*
        sub_structure->add_layer(
            (new LayerConfig("gamma_thalamus",
                IZHIKEVICH, thal_size, thal_size))
            ->set_property(IZ_INIT, "thalamo_cortical"));
        */

        // Cortical layers
        sub_structure->add_layer(
            (new LayerConfig("3_cortex",
                IZHIKEVICH, cortex_size, cortex_size,
                (new NoiseConfig("normal"))
                ->set_property("mean", std::to_string(cortex_noise))
                ->set_property("std_dev", std::to_string(cortex_noise_stdev))))
            ->set_property(IZ_INIT, "random positive"));
        /*
        sub_structure->add_layer(
            (new LayerConfig("6_cortex",
                IZHIKEVICH, cortex_size, cortex_size,
                (new NoiseConfig("normal"))
                ->set_property("mean", std::to_string(cortex_noise))
                ->set_property("std_dev", std::to_string(cortex_noise_stdev))))
            ->set_property(IZ_INIT, "regular"));
        */

        // Cortico-cortical connectivity
        /*
        sub_structure->connect_layers("3_cortex", "6_cortex",
            (new ConnectionConfig(exc_plastic, exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1, fraction)))
            ->set_arborized_config(new ArborizedConfig(intra_cortex_center,1,wrap)));
        */
        sub_structure->connect_layers("3_cortex", "3_cortex",
            (new ConnectionConfig(exc_plastic, exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1, fraction)))
            ->set_arborized_config(new ArborizedConfig(intra_cortex_center,1,wrap)));
        /*
        sub_structure->connect_layers("6_cortex", "6_cortex",
            (new ConnectionConfig(exc_plastic, exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1, fraction)))
            ->set_arborized_config(new ArborizedConfig(intra_cortex_center,1,wrap)));
        sub_structure->connect_layers("6_cortex", "3_cortex",
            (new ConnectionConfig(exc_plastic, exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1, fraction)))
            ->set_arborized_config(new ArborizedConfig(intra_cortex_center,1,wrap)));
        */

        if (intra_cortex_surround > intra_cortex_center) {
            /*
            sub_structure->connect_layers("3_cortex", "6_cortex",
                (new ConnectionConfig(false, inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(intra_cortex_center,
                        new FlatWeightConfig(0.1, fraction))))
                ->set_arborized_config(new ArborizedConfig(intra_cortex_surround,1,wrap)));
            */
            sub_structure->connect_layers("3_cortex", "3_cortex",
                (new ConnectionConfig(false, inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(intra_cortex_center,
                        new FlatWeightConfig(0.1, fraction))))
                ->set_arborized_config(new ArborizedConfig(intra_cortex_surround,1,wrap)));
            /*
            sub_structure->connect_layers("6_cortex", "6_cortex",
                (new ConnectionConfig(false, inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(intra_cortex_center,
                        new FlatWeightConfig(0.1, fraction))))
                ->set_arborized_config(new ArborizedConfig(intra_cortex_surround,1,wrap)));
            sub_structure->connect_layers("6_cortex", "3_cortex",
                (new ConnectionConfig(false, inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(intra_cortex_center,
                        new FlatWeightConfig(0.1, fraction))))
                ->set_arborized_config(new ArborizedConfig(intra_cortex_surround,1,wrap)));
            */
        }

        // Gamma connectivity
        /*
        sub_structure->connect_layers("gamma_thalamus", "6_cortex",
            (new ConnectionConfig(exc_plastic, 10 + exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1*thal_ratio, fraction)))
            ->set_arborized_config(new ArborizedConfig(gamma_center,1,wrap)));
        sub_structure->connect_layers("6_cortex", "gamma_thalamus",
            (new ConnectionConfig(exc_plastic, 10 + exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1*thal_ratio, fraction)))
            ->set_arborized_config(new ArborizedConfig(gamma_center,1,wrap)));

        if (gamma_surround > gamma_center) {
            sub_structure->connect_layers("gamma_thalamus", "6_cortex",
                (new ConnectionConfig(false, 10 + inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(gamma_center,
                        new FlatWeightConfig(0.1*thal_ratio, fraction))))
                ->set_arborized_config(new ArborizedConfig(gamma_surround,1,wrap)));
            sub_structure->connect_layers("6_cortex", "gamma_thalamus",
                (new ConnectionConfig(false, 10 + inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(gamma_center,
                        new FlatWeightConfig(0.1*thal_ratio, fraction))))
                ->set_arborized_config(new ArborizedConfig(gamma_surround,1,wrap)));
        }
        */

        // Feedforward pathway
        if (i > 0) {
            Structure::connect(sub_structures[i-1], "3_cortex",
                sub_structure, "3_cortex",
                (new ConnectionConfig(exc_plastic, 0, 0.5, CONVERGENT, ADD,
                    new FlatWeightConfig(0.1*thal_ratio, fraction)))
                ->set_arborized_config(new ArborizedConfig(ff_center,1,wrap)));
            Structure::connect(sub_structure, "3_cortex",
                sub_structures[i-1], "3_cortex",
                (new ConnectionConfig(exc_plastic, 0, 0.5, CONVERGENT, ADD,
                    new FlatWeightConfig(0.1*thal_ratio, fraction)))
                ->set_arborized_config(new ArborizedConfig(ff_center,1,wrap)));
            if (ff_center > ff_surround) {
                Structure::connect(sub_structures[i-1], "3_cortex",
                    sub_structure, "3_cortex",
                    (new ConnectionConfig(false, 0, 0.5, CONVERGENT, SUB,
                        new SurroundWeightConfig(ff_center,
                            new FlatWeightConfig(0.1*thal_ratio, fraction))))
                    ->set_arborized_config(new ArborizedConfig(ff_surround,1,wrap)));
                Structure::connect(sub_structure, "3_cortex",
                    sub_structures[i-1], "3_cortex",
                    (new ConnectionConfig(false, 0, 0.5, CONVERGENT, SUB,
                        new SurroundWeightConfig(ff_center,
                            new FlatWeightConfig(0.1*thal_ratio, fraction))))
                    ->set_arborized_config(new ArborizedConfig(ff_surround,1,wrap)));
            }
        }

        // Thalamocortical control connectivity
        Structure::connect(main_structure, "tl1_thalamus",
            sub_structure, "3_cortex",
            (new ConnectionConfig(false, 0, 0.5, FULLY_CONNECTED, MULT,
                new FlatWeightConfig(0.1*thal_ratio)))
            ->set_property("myelinated", "true"));
        /*
        Structure::connect(main_structure, "tl1_thalamus",
            sub_structure, "6_cortex",
            (new ConnectionConfig(false, 0, 0.5, FULLY_CONNECTED, MULT,
                new FlatWeightConfig(0.1*thal_ratio)))
            ->set_property("myelinated", "true"));
        */

        sub_structure->add_module("3_cortex", new ModuleConfig(output_name));
        //sub_structure->add_module("6_cortex", new ModuleConfig(output_name));
        //sub_structure->add_module("gamma_thalamus", new ModuleConfig(output_name));

        model->add_structure(sub_structure);
        sub_structures.push_back(sub_structure);
    }

    // Sensory relay to cortex
    Structure::connect(main_structure, "feedforward",
        sub_structures[0], "3_cortex",
        (new ConnectionConfig(exc_plastic, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1*thal_ratio, fraction)))
        ->set_arborized_config(new ArborizedConfig(ff_center,1,wrap)));
    Structure::connect(main_structure, "feedforward",
        sub_structures[0], "3_cortex",
        (new ConnectionConfig(false, 0, 0.5, CONVERGENT, SUB,
            new SurroundWeightConfig(ff_center,
                new FlatWeightConfig(0.1*thal_ratio, fraction))))
        ->set_arborized_config(new ArborizedConfig(ff_surround,1,wrap)));

    // Modules
    main_structure->add_module("tl1_thalamus",
        (new ModuleConfig("random_input"))
        ->set_property("max", "3")
        ->set_property("rate", "500")
        ->set_property("verbose", "true"));
    main_structure->add_module("feedforward",
        new ModuleConfig(output_name));

    Clock clock(true);
    //Clock clock(1.0f);
    print_model(model);
    delete clock.run(model, 500000, true, nullptr, false);
}


static void print_subset_overlap(Structure *structure) {
    for (auto layer : structure->get_layers()) {
        int *grid = (int*) calloc (layer->size, sizeof(int));
        for (auto conn : layer->get_input_connections()) {
            if (conn->type == SUBSET) {
                auto sc = conn->get_config()->get_subset_config();
                for (int row = sc->to_row_start; row < sc->to_row_end; ++row) {
                    for (int col = sc->to_col_start; col < sc->to_col_end; ++col) {
                        ++grid[row * layer->columns + col];
                    }
                }
            }
        }
        for (int row = 0; row < layer->rows; ++row) {
            for (int col = 0; col < layer->columns; ++col) {
                printf("%d ", grid[row * layer->columns + col]);
            }
            printf("\n");
        }
        free(grid);
    }
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(nullptr));

    // Suppress warnings
    ErrorManager::get_instance()->set_warnings(false);

    try {
        //mnist_test();
        //speech_train();
        //speech_test(std::string(argv[1]));
        //maze_game_test();
        //old_test();
        //simple_test();
        //single_field_test();
        //game_of_life_test();
        working_memory_test();

        return 0;
    } catch (std::runtime_error e) {
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
