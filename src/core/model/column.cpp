#include "model/column.h"

#define IZHIKEVICH "izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

/* Global shared variables */
static std::string learning_rate = "0.004";
static int intercortical_delay = 0;
static int inh_ratio = 2;

Column::Column(std::string name, int rows, int columns, bool plastic)
        : Structure(name, PARALLEL),
              cortex_rows(rows),
              cortex_columns(columns),

              inh_rows(cortex_rows / inh_ratio),
              inh_columns(cortex_columns / inh_ratio),

              exc_plastic(plastic),
              exc_inh_plastic(plastic) {
    /* Cortical Layers */
    //add_neural_field("3a");
    //add_neural_field("3");
    add_neural_field("4");
    //add_neural_field("5");

    /* Intracortical Connections */
    // One-Way
    //connect_fields_one_way("3", "4", 9, 0.1);
    //connect_fields_one_way("4", "5", 9, 0.1);
    //connect_fields_one_way("5", "4", 9, 0.1);
}

void Column::add_input(bool plastic, int num_symbols,
        std::string module_name, std::string module_params) {
    // Input layer
    this->add_layer((new LayerConfig("input", IZHIKEVICH, 1, num_symbols))
        ->set_property(IZ_INIT, "regular"));
    this->add_module("input", module_name, module_params);

    int num_tethers = 10;
    int tether_to_size = 5;
    int to_row_range = cortex_rows - tether_to_size;
    int to_col_range = cortex_columns - tether_to_size;

    // Add num_tethers tethers for each symbol
    for (int i = 0 ; i < num_symbols ; ++i) {
        printf("Tethers for symbol %d\n", i);
        for (int j = 0 ; j < num_tethers ; ++j) {
            int start_to_row = fRand(to_row_range);
            int start_to_col = fRand(to_col_range);
            printf("    (%4d, %4d)\n", start_to_row, start_to_col);

            connect_layers("input", "4_pos",
                (new ConnectionConfig(false, 0, 1, SUBSET, ADD,
                new FlatWeightConfig(0.5, 0.09)))
                ->set_subset_config(
                    new SubsetConfig(
                        0, 1,
                        i, i+1,
                        start_to_row, start_to_row + tether_to_size,
                        start_to_col, start_to_col + tether_to_size))
                ->set_property("myelinated", "true"));
        }
    }
}

void Column::add_lined_up_input(bool plastic, int num_symbols,
        std::string module_name, std::string module_params) {
    // Input layer
    this->add_layer((new LayerConfig("input", IZHIKEVICH, 1, num_symbols))
        ->set_property(IZ_INIT, "regular"));
    this->add_module("input", module_name, module_params);

    int num_tethers = 1;
    int tether_to_size = 3;
    int to_row_range = cortex_rows - tether_to_size;
    int to_col_range = cortex_columns - tether_to_size;

    int row = 0;

    // Add num_tethers tethers for each symbol
    for (int i = 0 ; i < num_symbols ; ++i) {
        printf("Tethers for symbol %d\n", i);
        for (int j = 0 ; j < num_tethers ; ++j) {
            printf("    (%4d, %4d)\n", row, 0);

            connect_layers("input", "4_pos",
                (new ConnectionConfig(false, 0, 1, SUBSET, ADD,
                new FlatWeightConfig(1.0, 1.0)))
                ->set_subset_config(
                    new SubsetConfig(
                        0, 1,
                        i, i+1,
                        row, row + tether_to_size,
                        0, tether_to_size))
                ->set_property("myelinated", "true"));
            row += 3;
        }
    }
}

void Column::connect(Column *col_a, Column *col_b,
        std::string name_a, std::string name_b,
        int num_tethers, int tether_from_size, int tether_to_size,
        float density) {
    int from_row_range = col_a->cortex_rows - tether_from_size;
    int from_col_range = col_a->cortex_columns - tether_from_size;
    int to_row_range = col_b->cortex_rows - tether_to_size;
    int to_col_range = col_b->cortex_columns - tether_to_size;

    for (int i = 0; i < num_tethers ; ++i) {
        int start_from_row = fRand(from_row_range);
        int start_from_col = fRand(from_col_range);
        int start_to_row = fRand(to_row_range);
        int start_to_col = fRand(to_col_range);

        Structure::connect(
            col_a, name_a, col_b, name_b,
            (new ConnectionConfig(
                col_b->exc_plastic, 0,
                0.5, SUBSET, ADD,
                new FlatWeightConfig(0.1, density)))
            ->set_subset_config(
                new SubsetConfig(
                    start_from_row, start_from_row + tether_from_size,
                    start_from_col, start_from_col + tether_from_size,
                    start_to_row, start_to_row + tether_to_size,
                    start_to_col, start_to_col + tether_to_size))
            ->set_property("learning rate", learning_rate)
            ->set_property("myelinated", "true"));
    }
}

void Column::add_neural_field(std::string field_name) {
    add_layer((new LayerConfig(field_name + "_pos",
        IZHIKEVICH, cortex_rows, cortex_columns))
            ->set_property(IZ_INIT, "random positive")
            //->set_property(IZ_INIT, "regular")
            ->set_property("spacing", "0.09"));

    add_layer((new LayerConfig(field_name + "_neg",
        IZHIKEVICH, inh_rows, inh_columns))
            ->set_property(IZ_INIT, "random negative")
            //->set_property(IZ_INIT, "fast")
            ->set_property("spacing", "0.18"));

    // Excitatory self connections
    int self_spread = 15;
    connect_layers(field_name + "_pos", field_name + "_pos",
        (new ConnectionConfig(
            exc_plastic, 1, 0.5, CONVERGENT, ADD,
            (new FlatWeightConfig(0.1, 0.09))
                ->set_diagonal(false)))
        ->set_arborized_config(
            new ArborizedConfig(self_spread, 1, -self_spread/2))
        ->set_property("learning rate", learning_rate));

    // Exc -> Inh
    int exc_inh_spread = 15;
    connect_layers(field_name + "_pos", field_name + "_neg",
        (new ConnectionConfig(
            exc_inh_plastic, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.09)))
        ->set_arborized_config(
            new ArborizedConfig(exc_inh_spread, inh_ratio, -exc_inh_spread/2))
        ->set_property("learning rate", learning_rate));

    // Inh -> Exc
    int inh_exc_spread = 15;
    connect_layers(field_name + "_neg", field_name + "_pos",
        (new ConnectionConfig(
            false, 0, 0.5, DIVERGENT, SUB,
            new FlatWeightConfig(1.0, 0.5)))
        ->set_arborized_config(
            new ArborizedConfig(inh_exc_spread, inh_ratio, -inh_exc_spread/2))
        ->set_property("learning rate", learning_rate));
}

void Column::connect_fields_one_way(std::string src, std::string dest,
        int spread, float density) {
    ConnectionConfig *config;
    if (spread == 1)
        config =
            (new ConnectionConfig(
                exc_plastic, 1, 0.5, ONE_TO_ONE, ADD,
                (new FlatWeightConfig(0.1, density))));
    else
        config =
            (new ConnectionConfig(
                exc_plastic, 1, 0.5, CONVERGENT, ADD,
                (new FlatWeightConfig(0.1, density))))
            ->set_arborized_config(
                new ArborizedConfig(spread, 1, -spread/2));

    connect_layers(src + "_pos", dest + "_pos",
        config->set_property("learning rate", learning_rate));
}
