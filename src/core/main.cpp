#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model/auditory_cortex.h"
#include "model/maze_cortex.h"
#include "model/model.h"
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
    structure->add_layer((new LayerConfig(
        "input_layer", model_name, 1, 10))
			->set_property(IZ_INIT, "default"));
    structure->add_layer((new LayerConfig(
        "exc_thalamus", model_name, resolution, resolution))
			->set_property(IZ_INIT, "thalamo_cortical"));
    structure->add_layer((new LayerConfig(
        "inh_thalamus", model_name, resolution, resolution))
			->set_property(IZ_INIT, "random negative"));
    structure->add_layer((new LayerConfig(
        "exc_cortex", model_name, resolution, resolution))
            ->set_property(IZ_INIT, "random positive"));
    structure->add_layer((new LayerConfig(
        "inh_cortex", model_name, resolution, resolution))
            ->set_property(IZ_INIT, "random negative"));

    /* Forward excitatory pathway */
    structure->connect_layers("input_layer", "exc_thalamus",
        (new ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(1, 0.01)))
        ->set_property("myelinated", "true"));
    structure->connect_layers("exc_thalamus", "exc_cortex",
        (new ConnectionConfig(true, 0, 10, CONVERGENT, ADD,
            new RandomWeightConfig(1, 0.1)))
        ->set_property("myelinated", "")
        ->set_arborized_config(new ArborizedConfig(15, 1, -7)));
    structure->connect_layers("exc_cortex", "exc_cortex",
        (new ConnectionConfig(true, 2, 5, CONVERGENT, ADD,
            new RandomWeightConfig(1, 0.1)))
        ->set_property("myelinated", "")
        ->set_arborized_config(new ArborizedConfig(31, 1, -15)));

    /* Cortical inhibitory loop */
    structure->connect_layers("exc_cortex", "inh_cortex",
        (new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.1)))
        ->set_property("myelinated", "")
        ->set_arborized_config(new ArborizedConfig(31, 1, -15)));
    structure->connect_layers("inh_cortex", "exc_cortex",
        (new ConnectionConfig(false, 0, 5, CONVERGENT, SUB,
            new RandomWeightConfig(1)))
        ->set_property("myelinated", "")
        ->set_arborized_config(new ArborizedConfig(5, 1, -2)));

    /* Cortico-thalamic inhibitory loop */
    structure->connect_layers("exc_cortex", "inh_thalamus",
        (new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new RandomWeightConfig(0.1)))
        ->set_property("myelinated", "")
        ->set_arborized_config(new ArborizedConfig(7, 1, -3)));
    structure->connect_layers("inh_thalamus", "exc_thalamus",
        (new ConnectionConfig(false, 0, 5, CONVERGENT, SUB,
            new FlatWeightConfig(1)))
        ->set_property("myelinated", "")
        ->set_arborized_config(new ArborizedConfig(5, 1, -2)));


    // Modules
    //std::string output_name = "dummy_output";
    std::string output_name = "visualizer_output";

    structure->add_module("input_layer",
        new ModuleConfig("random_input", "5 1000000"));
    structure->add_module("exc_thalamus",
        new ModuleConfig(output_name));
    structure->add_module("exc_cortex",
        new ModuleConfig(output_name));
    structure->add_module("exc_thalamus",
        new ModuleConfig("heatmap"));
    structure->add_module("exc_cortex",
        new ModuleConfig("heatmap"));
    //structure->add_module("inh_cortex", output_name, "8");
    //structure->add_module("inh_thalamus", output_name, "8");

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
    structure->add_layer((new LayerConfig(
        "hid_1", model_name, resolution, resolution))
			->set_property(IZ_INIT, "regular"));
    structure->add_layer((new LayerConfig(
        "hid_2", model_name, resolution, resolution))
			->set_property(IZ_INIT, "regular"));

    /* Forward excitatory pathway */
    structure->connect_layers("input_layer", "hid_1",
        (new ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(1, 0.05)))
        ->set_property("myelinated", "true"));

    structure->connect_layers("hid_1", "hid_2",
        (new ConnectionConfig(true, 10, 5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field, 1, -exc_field/2)));
    /*
    structure->connect_layers("hid_1", "hid_2",
        (new ConnectionConfig(false, 10, 5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field, 1, -inh_field/2)));
    */

    /* Recurrent self connectivity */
    structure->connect_layers("hid_1", "hid_1",
        (new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field, 1, -exc_field/2)));
    structure->connect_layers("hid_1", "hid_1",
        (new ConnectionConfig(false, 0, 5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field, 1, -inh_field/2)));

    structure->connect_layers("hid_2", "hid_2",
        (new ConnectionConfig(true, 0, 5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field, 1, -exc_field/2)));
    structure->connect_layers("hid_2", "hid_2",
        (new ConnectionConfig(false, 0, 5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field, 1, -inh_field/2)));

    /* Feedback connectivity */
    structure->connect_layers("hid_2", "hid_1",
        (new ConnectionConfig(true, 10, 5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field, 1, -exc_field/2)));
    /*
    structure->connect_layers("hid_2", "hid_1",
        (new ConnectionConfig(false, 10, 5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field, 1, -inh_field/2)));
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

    print_model(model);
    Clock clock(true);
    delete clock.run(model, 1000000, true);
    delete model;
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
        structure->add_layer((new LayerConfig(std::to_string(i),
            "leaky_izhikevich", 28, 28, 0.5))
                ->set_property(IZ_INIT, "regular"));
        structure->connect_layers("input_layer", std::to_string(i),
            (new ConnectionConfig(true, 0, 0.5, FULLY_CONNECTED, ADD,
                new FlatWeightConfig(0.1, 0.1)))
            ->set_arborized_config(new ArborizedConfig(9,1,-4)));

        structure->connect_layers(std::to_string(i), std::to_string(i),
            (new ConnectionConfig(true, 0, 0.5, FULLY_CONNECTED, ADD,
                new RandomWeightConfig(0.1, 0.1)))
            ->set_arborized_config(new ArborizedConfig(9,1,-4)));
        structure->connect_layers(std::to_string(i), std::to_string(i),
            (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, SUB,
                new FlatWeightConfig(0.5, 0.1)))
            ->set_arborized_config(new ArborizedConfig(11,1,-5)));
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

    auto state = clock.run(model, 10, true);

    MazeGame::get_instance(true)->set_board_dim(board_dim);
    state = clock.run(model, 10, true, state);
    delete state;

    //delete clock.run(model, 100, true);
    std::cout << "\n";

    delete model;
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
    ErrorManager::get_instance()->suppress_warnings();

    try {
        //mnist_test();
        //speech_train();
        //speech_test(std::string(argv[1]));
        //maze_game_test();
        //old_test();
        //simple_test();
        single_field_test();

        return 0;
    } catch (const char* msg) {
        printf("\n\nEXPECTED: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
