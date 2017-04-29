#include "model/column.h"

#define IZHIKEVICH "izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

/* Global shared variables */
static std::string base_conductance = "1.0";
static std::string learning_rate = "0.004";
static int intracortical_delay = 0;
static int intercortical_delay = 0;

static int inh_ratio = 2;

Column::Column(std::string name, int cortex_size, bool plastic)
        : Structure(name, PARALLEL),
              conductance(base_conductance),
              cortex_size(cortex_size),
              inh_size(cortex_size / inh_ratio),
              exc_plastic(plastic) {
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
        ->set_property("conductance", conductance)
        ->set_property("myelinated", "true"));
}

void Column::connect(Column *col_a, Column *col_b,
        std::string name_a, std::string name_b) {
    /*
    Structure::connect(
        col_a, name_a, col_b, name_b,
        (new ConnectionConfig(
            col_b->exc_plastic, intercortical_delay,
            max, FULLY_CONNECTED, ADD,
            new LogNormalWeightConfig(m, sd, f)))
        ->set_property("conductance", col_b->conductance)
        ->set_property("learning rate", lr));
    */
}

void Column::add_neural_field(std::string field_name) {
    add_layer((new LayerConfig(field_name + "_pos",
        IZHIKEVICH, cortex_size, cortex_size))
            //->set_property(IZ_INIT, "random positive"));
            ->set_property(IZ_INIT, "regular")
            ->set_property("spacing", "0.09"));

    add_layer((new LayerConfig(field_name + "_neg",
        IZHIKEVICH, inh_size, inh_size))
            //->set_property(IZ_INIT, "random negative"));
            ->set_property(IZ_INIT, "regular")
            ->set_property("spacing", "0.18"));

    // Excitatory self connections
    int self_spread = 18;
    connect_layers(field_name + "_pos", field_name + "_pos",
        (new ConnectionConfig(
            exc_plastic, 0, 0.5, CONVERGENT, ADD,
            (new FlatWeightConfig(0.1, 0.09))
                ->set_diagonal(false)))
        ->set_arborized_config(
            new ArborizedConfig(self_spread, 1, -self_spread/2))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));

    // Exc -> Inh
    int exc_inh_spread = 5;
    connect_layers(field_name + "_pos", field_name + "_neg",
        (new ConnectionConfig(
            false, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.09)))
        ->set_arborized_config(
            new ArborizedConfig(exc_inh_spread, inh_ratio, -exc_inh_spread/2))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));

    // Inh -> Exc
    int inh_exc_spread = 7;
    connect_layers(field_name + "_neg", field_name + "_pos",
        (new ConnectionConfig(
            false, 0, 0.5, DIVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.09)))
        ->set_arborized_config(
            new ArborizedConfig(inh_exc_spread, inh_ratio, -inh_exc_spread/2))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::connect_fields_one_way(std::string src, std::string dest) {
    /*
    connect_layers(
        src, dest,
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, one_way_max_weight,
            FULLY_CONNECTED, ADD,
            new LogNormalWeightConfig(
                one_way_mean, one_way_std_dev, one_way_fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
    */
}
