#include "model/maze_cortex.h"

#define IZHIKEVICH "izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

const std::string learning_rate = "0.004";

MazeCortex::MazeCortex(Model *model, int board_dim, int cell_size)
        : Structure("Maze Cortex", PARALLEL),
          board_dim(board_dim), cortex_size(board_dim*cell_size) {
    this->add_cortical_layer("3b");

    // Layer V projecting
    int v_spacing = 0.1 * cortex_size / 2;
    add_layer((new LayerConfig("5p_pos",
        IZHIKEVICH, 2, 2))
            ->set_property("spacing", std::to_string(v_spacing))
            ->set_property(IZ_INIT, "regular"));
    /*
    add_layer((new LayerConfig("5p_neg",
        IZHIKEVICH, 1, 1))
            ->set_property(IZ_INIT, "fast"));
    connect_layers("5p_pos", "5p_neg",
        (new ConnectionConfig(
            false, 1, 0.1, FULLY_CONNECTED, ADD,
            (new FlatWeightConfig(0.1, 1.0))))
        ->set_property("learning rate", learning_rate)
        ->set_property("myelinated", "true"));
    connect_layers("5p_neg", "5p_pos",
        (new ConnectionConfig(
            false, 1, 0.1, FULLY_CONNECTED, SUB,
            (new FlatWeightConfig(0.1, 1.0))))
        ->set_property("learning rate", learning_rate)
        ->set_property("myelinated", "true"));

    connect_layers("3b_pos", "5p_pos",
        (new ConnectionConfig(
            true, 1, 0.1, FULLY_CONNECTED, ADD,
            (new FlatWeightConfig(0.1, 0.1))))
        ->set_property("learning rate", learning_rate)
        ->set_property("myelinated", "true"));
    */

    int offset = v_spacing / 2;
    for (int i = 0 ; i < 2 ; ++i) {
        for (int j = 0 ; j < 2 ; ++j) {
            connect_layers("3b_pos", "5p_pos",
                (new ConnectionConfig(
                    true, 0, 0.5, SUBSET, ADD,
                    (new FlatWeightConfig(0.1, 0.1))))
                ->set_subset_config(
                    new SubsetConfig(
                        (i * cortex_size / 2), ((i+1) * cortex_size / 2),
                        (j * cortex_size / 2), ((j+1) * cortex_size / 2),
                        i, i+1,
                        j, j+1))
                ->set_property("x offset", std::to_string(offset))
                ->set_property("y offset", std::to_string(offset))
                ->set_property("learning rate", learning_rate));
            printf("(%d %d, %d %d) => (%d, %d)\n",
                (i * cortex_size / 2), ((i+1) * cortex_size / 2),
                (j * cortex_size / 2), ((j+1) * cortex_size / 2),
                i, j);
        }
    }

    add_module("5p_pos", "maze_output");

    this->add_input_random("3b_pos", "player_input", "maze_input", "player");
    this->add_input_random("3b_pos", "goal_input", "maze_input", "goal");

    // Dopamine
    add_layer(
        (new LayerConfig("dopamine", IZHIKEVICH, 1, 1))
        ->set_property(IZ_INIT, "regular"));
    add_module("dopamine", "maze_input", "reward");
    connect_layers(
        "dopamine",
        "3b_pos",
        (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, REWARD,
            new FlatWeightConfig(1.0, 1.0)))
        ->set_property("myelinated", "true"));

    model->add_structure(this);
}

void MazeCortex::add_cortical_layer(std::string name) {
    add_layer((new LayerConfig(name + "_pos",
        IZHIKEVICH, cortex_size, cortex_size))
            ->set_property(IZ_INIT, "random positive")
            //->set_property(IZ_INIT, "regular")
            ->set_property("spacing", "0.1"));

    add_layer((new LayerConfig(name + "_neg",
        IZHIKEVICH, cortex_size / 2, cortex_size / 2))
            ->set_property(IZ_INIT, "random negative")
            //->set_property(IZ_INIT, "fast")
            ->set_property("spacing", "0.2"));

    // Excitatory self connections
    int self_spread = 32;
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
            new FlatWeightConfig(0.1, 0.1)))
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

void MazeCortex::connect_one_way(std::string name1, std::string name2) {
    int spread = 32;
    connect_layers(name1, name2,
        (new ConnectionConfig(
            true, 1, 0.5, CONVERGENT, ADD,
            (new FlatWeightConfig(0.1, 0.1))))
        ->set_arborized_config(
            new ArborizedConfig(spread, 1, -spread/2))
        ->set_property("learning rate", learning_rate));
}

void MazeCortex::add_input_grid(std::string layer, std::string input_name,
        std::string module_name, std::string module_params) {
    add_layer(
        (new LayerConfig(input_name, IZHIKEVICH, 1, board_dim*board_dim))
        ->set_property(IZ_INIT, "regular"));
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
        ->set_property(IZ_INIT, "regular"));
    add_module(input_name, module_name, module_params);

    int num_tethers = 1; // 3;
    int spread = 32;

    int to_row_range = cortex_size - spread;
    int to_col_range = cortex_size - spread;

    // Add num_tethers tethers for each symbol
    for (int i = 0 ; i < board_dim*board_dim ; ++i) {
        printf("Tethers for symbol %d\n", i);
        for (int j = 0 ; j < num_tethers ; ++j) {
            int start_to_row = fRand(to_row_range);
            int start_to_col = fRand(to_col_range);
            printf("    (%4d, %4d)\n", start_to_row, start_to_col);

            connect_layers(input_name, layer,
                //(new ConnectionConfig(false, 0, 1, SUBSET, ADD,
                (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, ADD,
                new FlatWeightConfig(1.0, 0.1)))
                ->set_subset_config(
                    new SubsetConfig(
                        0, 1,
                        i, i+1,
                        start_to_row, start_to_row + spread,
                        start_to_col, start_to_col + spread))
                ->set_property("myelinated", "true"));
        }
    }
}
