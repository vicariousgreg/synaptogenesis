#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "model/auditory_cortex.h"
#include "model/maze_cortex.h"
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
    delete clock.run(model, iterations, verbose);

    // Benchmark the network
    // Use max refresh rate possible
    // Run for 100 iterations
    //Clock clock(false);  // No refresh rate synchronization
    //delete clock.run(model, 100, verbose);
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

    AuditoryCortex *auditory_cortex = new AuditoryCortex(model, 41, 3);
    auditory_cortex->add_input("3b_pos", "speech_input", "csv_input", "./resources/speech.csv 0 1 0.25");
    auditory_cortex->add_module_all("visualizer_output", "");
    auditory_cortex->add_module_all("heatmap", "");

    // Modules
    //structure->add_module("convergent_layer", "csv_output");

    std::cout << "Speech test......\n";
    print_model(model);
    //Clock clock(60.0f);
    Clock clock(true);
    auto state = clock.run(model, 1000000, true);
    //state->transfer_to_host();
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
    maze_cortex->add_module_all("visualizer_output", "");
    maze_cortex->add_module_all("heatmap", "");

    std::cout << "Maze game test......\n";
    print_model(model);
    Clock clock(true);
    //Clock clock(10.0f);
    delete clock.run(model, 1000000, true);
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
        speech_test();
        //maze_game_test();

        return 0;
    } catch (const char* msg) {
        printf("\n\nEXPECTED: %s\n", msg);
        printf("Fatal error -- exiting...\n");
        return 1;
    }
}
