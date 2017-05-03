#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model/cortical_region.h"
#include "model/sensory_cortex.h"
#include "model/motor_cortex.h"
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
            (new ConnectionConfig(true, 0, 5, CONVOLUTIONAL, ADD,
                new RandomWeightConfig(5)))
            ->set_arborized_config(new ArborizedConfig(5,1)));

        structure->connect_layers(std::to_string(i), std::to_string(i),
            (new ConnectionConfig(true, 0, 1, CONVOLUTIONAL, ADD,
                new RandomWeightConfig(0.1)))
            ->set_arborized_config(new ArborizedConfig(5,1)));
        structure->connect_layers(std::to_string(i), std::to_string(i),
            (new ConnectionConfig(false, 0, 2, CONVOLUTIONAL, DIV,
                new RandomWeightConfig(2)))
            ->set_arborized_config(new ArborizedConfig(7,1)));
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

    // Intermediate cortical layers
    int cortex_rows = 128;
    int cortex_columns = 64;
    int num_symbols = 41;

    Column *sensory_column = new Column("sensory", cortex_rows, cortex_columns, true);
    sensory_column->add_lined_up_input(
        false, num_symbols, "csv_input", "./resources/speech.csv 0 1 0.25");
    sensory_column->add_module_all("visualizer_output", "");
    sensory_column->add_module_all("heatmap", "");
    model->add_structure(sensory_column);

    Column *column1 = new Column("column1", cortex_rows, cortex_columns, true);
    column1->add_module_all("visualizer_output", "");
    column1->add_module_all("heatmap", "");
    model->add_structure(column1);

    Column::connect(
        sensory_column, column1,
        "4_pos", "4_pos",
        100, 5, 5,
        0.09);

    // Modules
    //structure->add_module("convergent_layer", "csv_output");

    std::cout << "Speech test......\n";
    print_model(model);
    //Clock clock(60.0f);
    Clock clock(true);
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

void symbol_test() {
    /* Construct the model */
    Model *model = new Model();

    // Intermediate cortical layers
    int cortex_rows = 64;
    int cortex_columns = 64;
    int num_symbols = 5;

    Column *sensory_column = new Column("sensory", cortex_rows, cortex_columns, true);
    sensory_column->add_input(true, num_symbols, "one_hot_cyclic_input", "3.78 100000");
    sensory_column->add_module_all("visualizer_output", "");
    sensory_column->add_module_all("heatmap", "");
    model->add_structure(sensory_column);

    Column *prev_column = sensory_column;
    std::vector<Column*> columns;
    int num_columns = 4;
    for (int i = 0 ; i < num_columns ; ++i) {
        Column *column = new Column("col" + std::to_string(i), cortex_rows, cortex_columns, true);
        column->add_module_all("visualizer_output", "");
        column->add_module_all("heatmap", "");

        model->add_structure(column);
        columns.push_back(column);

        Column::connect(
            prev_column, column,
            "4_pos", "4_pos",
            10, 10, 10,
            0.09);
        Column::connect(
            column, prev_column,
            "4_pos", "4_pos",
            10, 10, 10,
            0.09);
        prev_column = column;
    }

    for (auto column : columns)
        print_subset_overlap(column);

    std::cout << "Symbol test......\n";
    print_model(model);
    Clock clock(true);
    //Clock clock(100.0f);
    //Clock clock(10.0f);
    clock.run(model, 100000, true);
    std::cout << "\n";

    delete model;
}

void cortex_test() {
    /* Construct the model */
    Model *model = new Model();

    SensoryCortex *sensory =
        new SensoryCortex(model, true, 9, 32, 32);
    CorticalRegion *association =
        new CorticalRegion(model, "association", true, 1, 64, 64);
    MotorCortex *motor =
        new MotorCortex(model, true, 4, 32, 32);

    sensory->add_input("player_input", false, 9, "maze_input", "player");
    sensory->add_input("goal_input", false, 9, "maze_input", "goal");
    motor->add_output("motor_output", false, 4, "maze_output");

    sensory->add_module_all("visualizer_output");
    sensory->add_module_all("heatmap");
    association->add_module_all("visualizer_output");
    association->add_module_all("heatmap");
    motor->add_module_all("visualizer_output");
    motor->add_module_all("heatmap");

    // Feedforward
    sensory->connect(association,
        "5_pos", "4_pos",
        5, 5, 5, 1.0);
    association->connect(motor,
        "5_pos", "4_pos",
        5, 5, 5, 1.0);

    // Feedback
    /*
    motor->connect(association,
        "5_pos", "3_pos",
        5, 5, 5, 1.0);
    association->connect(sensory,
        "5_pos", "3_pos",
        5, 5, 5, 1.0);
    */

    // Diffuse dopaminergic input
    Structure *brainstem = new Structure("brainstem", PARALLEL);
    brainstem->add_layer(
        (new LayerConfig("dopamine", IZHIKEVICH, 1, 1))
        ->set_property(IZ_INIT, "regular"));
    brainstem->add_module("dopamine", "visualizer_output");
    brainstem->add_module("dopamine", "maze_input", "reward");
    //brainstem->add_module("dopamine", "print_output", "8");
    model->add_structure(brainstem);

    sensory->connect_diffuse(brainstem, "dopamine", REWARD, 1.0);
    association->connect_diffuse(brainstem, "dopamine", REWARD, 1.0);
    motor->connect_diffuse(brainstem, "dopamine", REWARD, 1.0);

    std::cout << "Cortex test......\n";
    print_model(model);
    Clock clock(true);
    //Clock clock(100.0f);
    //Clock clock(10.0f);
    clock.run(model, 100000, true);
    std::cout << "\n";

    delete model;
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(nullptr));

    // Suppress warnings
    ErrorManager::get_instance()->suppress_warnings();

    try {
        //mnist_test();
        //speech_test();
        //maze_game_test();
        //symbol_test();
        cortex_test();

        return 0;
    } catch (const char* msg) {
        printf("\n\nEXPECTED: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
