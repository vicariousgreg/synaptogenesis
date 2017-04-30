#include "model/column.h"

#define IZHIKEVICH "izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

/* Global shared variables */
static std::string learning_rate = "0.4";
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
    add_neural_field("4");

    /* Intracortical Connections */
    // One-Way
    //connect_fields_one_way("4", "3a");
}

void Column::add_input(bool plastic, int num_symbols,
        std::string module_name, std::string module_params) {
    // Input layer
    this->add_layer((new LayerConfig("input", IZHIKEVICH, 1, num_symbols))
        ->set_property(IZ_INIT, "regular"));
    this->add_module("input", module_name, module_params);

    // Input connection
    connect_layers("input", "4_pos",
        (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, ADD,
        new FlatWeightConfig(0.5, 0.01)))
        ->set_property("myelinated", "true"));
}

void Column::connect(Column *col_a, Column *col_b,
        std::string name_a, std::string name_b,
        int num_tethers, int tether_from_size, int tether_to_size) {
    int to_row_range = col_b->cortex_rows - tether_to_size;
    int to_col_range = col_b->cortex_columns - tether_to_size;
    int from_row_range = col_b->cortex_rows - tether_from_size;
    int from_col_range = col_b->cortex_columns - tether_from_size;

    for (int i = 0; i < num_tethers ; ++i) {
        int start_from_row = fRand(from_row_range);
        int start_from_col = fRand(from_col_range);
        int start_to_row = fRand(to_row_range);
        int start_to_col = fRand(to_col_range);

        Structure::connect(
            col_a, name_a, col_b, name_b,
            (new ConnectionConfig(
                col_b->exc_plastic, 0,
                0.5, FULLY_CONNECTED, ADD,
                new FlatWeightConfig(0.1, 0.09)))
            ->set_fully_connected_config(
                new FullyConnectedConfig(
                    start_from_row, start_from_row + tether_from_size,
                    start_from_col, start_from_col + tether_from_size,
                    start_to_row, start_to_row + tether_to_size,
                    start_to_col, start_to_col + tether_to_size
                ))
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
    int self_spread = 29;
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
            new FlatWeightConfig(1.0, 0.09)))
        ->set_arborized_config(
            new ArborizedConfig(inh_exc_spread, inh_ratio, -inh_exc_spread/2))
        ->set_property("learning rate", learning_rate));
}

void Column::connect_fields_one_way(std::string src, std::string dest) {
}
