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

Structure *add_column(Model *model, std::string name, std::string output_name, int size,
        float exc_noise_mean, float exc_noise_std_dev,
        float thal_noise_mean, float thal_noise_std_dev,
        bool exc_plastic, bool inh_plastic) {
    Structure *structure = model->add_structure(name);

    std::string pos_name = std::string("pos");
    std::string neg_name = std::string("neg");

    /*******************************************/
    /***************** CORTEX ******************/
    /*******************************************/
    // Ratio of exc:inh size
    int inh_ratio = 2;
    int inh_size = size / inh_ratio;
    int inh_factor = inh_ratio * inh_ratio;

    std::vector<std::string> stack_names;
    //stack_names.push_back("2_");
    stack_names.push_back("3a_");
    stack_names.push_back("4_");
    stack_names.push_back("56_");
    stack_names.push_back("6t_");

    // Cortical exc-inh layer pairs
    for (auto stack_name : stack_names) {
        structure->add_layer((new LayerConfig(stack_name + pos_name,
            IZHIKEVICH, size, size, exc_noise_mean, exc_noise_std_dev))
                ->set_property(IZ_INIT, "random positive"));
        structure->add_layer((new LayerConfig(stack_name + neg_name,
            IZHIKEVICH, inh_size, inh_size, 0.0, 0.0))
                ->set_property(IZ_INIT, "random negative"));

        // Excitatory -> Inhibitory Connection
        structure->connect_layers(stack_name + pos_name, stack_name + neg_name,
            new ConnectionConfig(false, 0, 4, FULLY_CONNECTED, ADD,
                new GaussianWeightConfig(1, 0.3, 0.1)));

        // Inhibitory -> Excitatory Connection
        structure->connect_layers(stack_name + neg_name, stack_name + pos_name,
            new ConnectionConfig(inh_plastic, 0, inh_factor*4, FULLY_CONNECTED, SUB,
                new GaussianWeightConfig(inh_factor*1, inh_factor*0.3, 0.1)));

        // Output modules
        if (output_name != "") {
            structure->add_module(stack_name + pos_name, output_name, "");
            structure->add_module(stack_name + neg_name, output_name, "");
        }
    }

    /*******************************************/
    /************** INTRACORTICAL **************/
    /*******************************************/
    int spread_ratio = 4;
    int spread = size / spread_ratio;
    int delay = 0;
    float max_weight = spread_ratio;

    std::map<std::string, std::vector<std::string> > one_way;
    //one_way["2_pos"].push_back("3a_pos");
    one_way["4_pos"].push_back("3a_pos");
    one_way["4_pos"].push_back("56_pos");
    one_way["3a_pos"].push_back("6t_pos");

    std::map<std::string, std::vector<std::string> > reentry;
    reentry["3a_pos"].push_back("56_pos");
    one_way["56_pos"].push_back("6t_pos");

    float fraction = 0.05;
    float mean = 1.0 * spread_ratio / fraction;
    float std_dev = 0.3 * spread_ratio / fraction;
    for (auto pair : one_way) {
        auto src = pair.first;
        for (auto dest : pair.second) {
            structure->connect_layers(
                src, dest,
                new ConnectionConfig(exc_plastic, delay, max_weight, CONVERGENT, ADD,
                    new GaussianWeightConfig(mean, std_dev, fraction),
                    new ArborizedConfig(spread,1,-spread/2)));
        }
    }

    mean = 0.05 * spread_ratio;
    std_dev = 0.01 * spread_ratio;
    fraction = 1.0;
    for (auto pair : reentry) {
        auto src = pair.first;
        for (auto dest : pair.second) {
            structure->connect_layers(
                src, dest,
                new ConnectionConfig(exc_plastic, delay, max_weight, CONVERGENT, ADD,
                    new GaussianWeightConfig(mean, std_dev, fraction),
                    new ArborizedConfig(spread,1,-spread/2)));
            structure->connect_layers(
                dest, src,
                new ConnectionConfig(exc_plastic, delay, max_weight, CONVERGENT, ADD,
                    new GaussianWeightConfig(mean, std_dev, fraction),
                    new ArborizedConfig(spread,1,-spread/2)));
        }
    }

    /*******************************************/
    /***************** THALAMUS ****************/
    /*******************************************/
    // Ratio of exc:thalamus size
    int thal_ratio = 2;
    int thal_size = size / thal_ratio;
    int thal_factor = thal_ratio * thal_ratio;

    // Thalamic exc-inh pair
    std::string thal_name = "thal_";
    structure->add_layer((new LayerConfig(thal_name + pos_name,
        IZHIKEVICH, thal_size, thal_size, thal_noise_mean, thal_noise_std_dev))
            ->set_property(IZ_INIT, "thalamo_cortical"));
    structure->add_layer((new LayerConfig(thal_name + neg_name,
        IZHIKEVICH, thal_size, thal_size, 0.0, 0.0))
            ->set_property(IZ_INIT, "thalamo_cortical"));

    // Excitatory -> Inhibitory Connection
    structure->connect_layers(thal_name + pos_name, thal_name + neg_name,
        new ConnectionConfig(false, 0, 4, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(1, 0.3, 0.1)));

    // Inhibitory -> Excitatory Connection
    structure->connect_layers(thal_name + neg_name, thal_name + pos_name,
        new ConnectionConfig(inh_plastic, 0, 4, FULLY_CONNECTED, SUB,
            new GaussianWeightConfig(1, 0.3, 0.1)));

    if (output_name != "") {
        structure->add_module(thal_name + pos_name, output_name, "");
        structure->add_module(thal_name + neg_name, output_name, "");
    }

    /*******************************************/
    /************ THALAMO-CORTICAL *************/
    /*******************************************/
    spread_ratio /= 2;
    spread = thal_size / spread_ratio;
    delay = 10;
    max_weight = spread_ratio;

    mean = 0.05 * spread_ratio;
    std_dev = 0.01 * spread_ratio;
    fraction = 1.0;
    std::string src = "6t_pos";
    std::string dest = "thal_pos";
    structure->connect_layers(
        src, dest,
        new ConnectionConfig(exc_plastic, delay, max_weight, CONVERGENT, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction),
            new ArborizedConfig(spread,thal_ratio,-spread/2)));
    structure->connect_layers(
        dest, src,
        new ConnectionConfig(exc_plastic, delay, max_weight, DIVERGENT, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction),
            new ArborizedConfig(spread,thal_ratio,-spread/2)));

    return structure;
}

void connect_columns(Structure *col_a, Structure *col_b,
        std::string name_a, std::string name_b,
        float mean, float std_dev, float fraction, float max,
        bool plastic) {
    Structure::connect(
        col_a, name_a, col_b, name_b,
        new ConnectionConfig(plastic, 2, max, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)));
}

void symbol_test() {
    /* Construct the model */
    Model *model = new Model();
    Structure *base = model->add_structure("base");

    std::string output_name = "visualizer_output";

    // Input layers
    base->add_layer(new LayerConfig("input1", RELAY, 1, 2));
    base->add_module("input1", "one_hot_cyclic_input", "1 12000");
    base->add_module("input1", output_name, "");

    base->add_layer(new LayerConfig("input2", RELAY, 1, 2));
    base->add_module("input2", "one_hot_cyclic_input", "1 10000 10000");
    base->add_module("input2", output_name, "");


    // Intermediate cortical layers
    int cortex_size = 32;
    float exc_noise_mean = 1.0;
    float exc_noise_std_dev = 0.1;
    float thal_noise_mean = 0.0;
    float thal_noise_std_dev = 0.0;
    bool exc_plastic = true;
    bool inh_plastic = true;
    Structure *column1 = add_column(model, "col1", output_name, cortex_size,
        exc_noise_mean, exc_noise_std_dev,
        thal_noise_mean, thal_noise_std_dev,
        exc_plastic, inh_plastic);
    Structure *column2 = add_column(model, "col2", output_name, cortex_size,
        exc_noise_mean, exc_noise_std_dev,
        thal_noise_mean, thal_noise_std_dev,
        exc_plastic, inh_plastic);

    // Input connections
    float input_strength = 25;
    Structure::connect(
        base, "input1",
        column1, "4_pos",
        new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(input_strength, input_strength/10, 0.025)));
    Structure::connect(
        base, "input2",
        column2, "4_pos",
        new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(input_strength, input_strength/10, 0.025)));

    // Intercortical connections
    // 1 <-> 3
    float mean = 0.0;
    float std_dev = 0.0;
    float fraction = 1.0;
    float max = 1;
    connect_columns(
        column1, column2,
        "56_pos", "56_pos",
        mean, std_dev, fraction, max,
        exc_plastic);
    connect_columns(
        column2, column1,
        "56_pos", "56_pos",
        mean, std_dev, fraction, max,
        exc_plastic);

    std::cout << "Symbol test......\n";
    print_model(model);
    Clock clock(true);
    clock.run(model, 1000000, true);
    std::cout << "\n";

    delete model;
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(nullptr));

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
