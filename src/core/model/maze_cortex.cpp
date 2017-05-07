#include "model/maze_cortex.h"

#define IZHIKEVICH "izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

const std::string learning_rate = "0.01";

MazeCortex::MazeCortex(Model *model, int board_dim, int cell_size)
        : Structure("Maze Cortex", PARALLEL),
          board_dim(board_dim), cell_size(cell_size),
          cortex_size(board_dim*cell_size) {
    this->add_cortical_layer("3b", 1);
    this->add_cortical_layer("56", 1);
    this->connect_one_way("3b_pos", "56_pos", 15, 0.5, 0, 1);
    //this->connect_one_way("56_pos", "3b_pos", 31, 0.5, 0, 1);

    // Basal Ganglia
    int bg_ratio = 2;
    int bg_spread = 31;
    this->add_cortical_layer("striatum", bg_ratio);

    connect_layers("56_pos", "striatum_pos",
        (new ConnectionConfig(
            true, 0, 0.5, CONVERGENT, ADD,
            (new FlatWeightConfig(0.01, 0.1))))
        ->set_arborized_config(
            new ArborizedConfig(bg_spread, bg_ratio, -bg_spread/2))
        ->set_property("myelinated", "true")
        ->set_property("learning rate", learning_rate));

    // Output Neurons
    add_layer((new LayerConfig("output",
        IZHIKEVICH, 2, 2))
            ->set_property(IZ_INIT, "bursting"));

    connect_layers("striatum_pos", "output",
        (new ConnectionConfig(
            true, 0, 0.5, FULLY_CONNECTED, ADD,
            (new FlatWeightConfig(0.1, 0.05))))
        ->set_property("myelinated", "true")
        ->set_property("learning rate", learning_rate));

    connect_layers("output", "striatum_neg",
        (new ConnectionConfig(
            false, 0, 0.5, FULLY_CONNECTED, ADD,
            (new FlatWeightConfig(0.5, 0.1))))
        ->set_property("myelinated", "true")
        ->set_property("learning rate", learning_rate));

    add_module("output", "maze_output");

    this->add_input_random("3b_pos", "player_input", "maze_input", "player");
    this->add_input_random("3b_pos", "goal_input", "maze_input", "goal");

    // Dopamine
    add_layer(
        (new LayerConfig("dopamine", IZHIKEVICH, 1, 1))
        ->set_property(IZ_INIT, "bursting"));
    add_module("dopamine", "maze_input", "reward");
    connect_layers(
        "dopamine",
        "striatum_pos",
        (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, REWARD,
            new FlatWeightConfig(1.0, 1.0)))
        ->set_property("myelinated", "true"));

    // Acetylcholine
    add_layer(
        (new LayerConfig("acetylcholine", IZHIKEVICH, 1, 1))
        ->set_property(IZ_INIT, "bursting"));
    add_module("acetylcholine", "maze_input", "modulate");
    connect_layers(
        "acetylcholine",
        "output",
        (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, MODULATE,
            new FlatWeightConfig(0.1, 1.0)))
        ->set_property("myelinated", "true"));
    connect_layers(
        "acetylcholine",
        "striatum_pos",
        (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, MODULATE,
            new FlatWeightConfig(0.1, 1.0)))
        ->set_property("myelinated", "true"));
    connect_layers(
        "acetylcholine",
        "56_pos",
        (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, MODULATE,
            new FlatWeightConfig(0.1, 1.0)))
        ->set_property("myelinated", "true"));

    model->add_structure(this);
}

void MazeCortex::add_cortical_layer(std::string name, int size_fraction) {
    int inh_ratio = 2;

    int exc_size = cortex_size / size_fraction;
    int inh_size = exc_size / inh_ratio;

    float exc_spacing = 0.1 * size_fraction;
    float inh_spacing = 0.1 * inh_ratio *  size_fraction;

    // Add layers
    add_layer((new LayerConfig(name + "_pos",
        IZHIKEVICH, exc_size, exc_size))
            ->set_property(IZ_INIT, "random positive")
            ->set_property("spacing", std::to_string(exc_spacing)));

    add_layer((new LayerConfig(name + "_neg",
        IZHIKEVICH, inh_size, inh_size))
            ->set_property(IZ_INIT, "random negative")
            ->set_property("spacing", std::to_string(inh_spacing)));

    // Excitatory self connections
    int self_spread = 31;
    connect_layers(name + "_pos", name + "_pos",
        (new ConnectionConfig(
            true, 2, 0.5, CONVERGENT, ADD,
            (new FlatWeightConfig(0.1, 0.1))
                ->set_diagonal(false)))
        ->set_arborized_config(
            new ArborizedConfig(self_spread, 1, -self_spread/2))
        ->set_property("learning rate", learning_rate));

    // Exc -> Inh
    int exc_inh_spread = 7;
    connect_layers(name + "_pos", name + "_neg",
        (new ConnectionConfig(
            true, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.25)))
        ->set_arborized_config(
            new ArborizedConfig(exc_inh_spread, 2, -exc_inh_spread/2))
        ->set_property("learning rate", learning_rate));

    // Inh -> Exc
    int inh_exc_spread = 7;
    connect_layers(name + "_neg", name + "_pos",
        (new ConnectionConfig(
            false, 0, 0.5, DIVERGENT, SUB,
            new FlatWeightConfig(1.0, 0.5)))
        ->set_arborized_config(
            new ArborizedConfig(inh_exc_spread, 2, -inh_exc_spread/2))
        ->set_property("learning rate", learning_rate));
}

void MazeCortex::connect_one_way(std::string name1, std::string name2,
        int spread, float fraction, int delay, int stride) {
    connect_layers(name1, name2,
        (new ConnectionConfig(
            true, delay, 0.5, CONVERGENT, ADD,
            (new FlatWeightConfig(0.1, fraction))))
        ->set_arborized_config(
            new ArborizedConfig(spread, stride, -spread/2))
        ->set_property("learning rate", learning_rate));
}

void MazeCortex::add_input_grid(std::string layer, std::string input_name,
        std::string module_name, std::string module_params) {
    add_layer(
        (new LayerConfig(input_name, IZHIKEVICH, 1, board_dim*board_dim))
        ->set_property(IZ_INIT, "bursting"));
    add_module(input_name, module_name, module_params);

    int spread = cortex_size / board_dim;

    for (int i = 0 ; i < board_dim; ++i) {
        for (int j = 0 ; j < board_dim; ++j) {
            int input_index = i * board_dim + j;

            connect_layers(
                input_name, layer,
                (new ConnectionConfig(false, 0, 1, SUBSET, ADD,
                    new FlatWeightConfig(1.0, 0.1)))
                ->set_subset_config(
                    new SubsetConfig(
                        0, 1,
                        input_index, input_index+1,
                        i * spread, (i+1) * spread,
                        j * spread, (j+1) * spread))
                ->set_property("myelinated", "true"));
        }
    }
}

void MazeCortex::add_input_random(std::string layer, std::string input_name,
        std::string module_name, std::string module_params) {
    // Input layer
    add_layer(
        (new LayerConfig(input_name, IZHIKEVICH, 1, board_dim*board_dim))
        ->set_property(IZ_INIT, "bursting"));
    add_module(input_name, module_name, module_params);

    int num_tethers = 1; // 3;
    int spread = cell_size * 1.5;

    int to_row_range = cortex_size - spread;
    int to_col_range = cortex_size - spread;

    // Add num_tethers tethers for each symbol
    for (int i = 0 ; i < board_dim*board_dim ; ++i) {
        //printf("Tethers for symbol %d\n", i);
        for (int j = 0 ; j < num_tethers ; ++j) {
            int start_to_row = fRand(to_row_range);
            int start_to_col = fRand(to_col_range);
            //printf("    (%4d, %4d)\n", start_to_row, start_to_col);

            connect_layers(input_name, layer,
                //(new ConnectionConfig(false, 0, 1, SUBSET, ADD,
                (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, ADD,
                //new FlatWeightConfig(0.5, 0.1)))
                new GaussianWeightConfig(0.5, 0.15, 0.001)))
                ->set_subset_config(
                    new SubsetConfig(
                        0, 1,
                        i, i+1,
                        start_to_row, start_to_row + spread,
                        start_to_col, start_to_col + spread))
                ->set_property("myelinated", "true")
                ->set_property("short term plasticity", "false"));
        }
    }
}
