#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model/model.h"
#include "model/column.h"
#include "model/weight_config.h"
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

void mnist_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = new Structure("mnist");
    model->add_structure(structure);

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
    Structure *structure = new Structure("speech", SEQUENTIAL);
    model->add_structure(structure);

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

void maze_game_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *structure = new Structure("maze_game");
    model->add_structure(structure);

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
    std::string output_name = "visualizer_output";

    // Intermediate cortical layers
    int cortex_size = 32;
    int num_symbols = 5;

    Column *column1 = new Column("col1", cortex_size);
    column1->add_input(num_symbols, "one_hot_cyclic_input", "1 10000");
    column1->add_module_all(output_name, "");
    model->add_structure(column1);

    /*
    Column *column2 = new Column("col2", cortex_size);
    column2->add_input(num_symbols, "one_hot_cyclic_input", "1 100000");
    column2->add_module_all(output_name, "");
    model->add_structure(column2);

    // Intercortical connections
    Column::connect(
        column1, column2,
        "56_pos", "56_pos");
    Column::connect(
        column2, column1,
        "56_pos", "56_pos");
    */

    std::cout << "Symbol test......\n";
    print_model(model);
    Clock clock(true);
    //Clock clock(10.0f);
    clock.run(model, 10000000, true);
    std::cout << "\n";

    delete model;
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(nullptr));

    // Suppress warnings
    //ErrorManager::get_instance()->suppress_warnings();

    try {
        //mnist_test();
        //speech_test();
        //maze_game_test();
        symbol_test();

        return 0;
    } catch (const char* msg) {
        printf("\n\nEXPECTED: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
