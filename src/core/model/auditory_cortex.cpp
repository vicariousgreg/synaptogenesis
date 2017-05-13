#include "model/auditory_cortex.h"

#define IZHIKEVICH "izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

const std::string learning_rate = "0.01";

AuditoryCortex::AuditoryCortex(Model *model, int spec_size, int spec_spread)
        : Structure("Auditory Cortex", PARALLEL),
          spec_size(spec_size), spec_spread(spec_spread),
          cortex_rows(spec_size*spec_spread),
          cortex_cols(cortex_rows*2) {
    this->add_cortical_layer("3b", 1);
    this->add_cortical_layer("5a", 1);
    this->connect_one_way("3b_pos", "5a_pos", 1, 1.0, 4);

    model->add_structure(this);
}

void AuditoryCortex::add_cortical_layer(std::string name, int size_fraction) {
    int inh_ratio = 2;

    int exc_rows = cortex_rows / size_fraction;
    int exc_cols = cortex_cols / size_fraction;
    int inh_rows = exc_rows / inh_ratio;
    int inh_cols = exc_cols / inh_ratio;

    float exc_spacing = 0.1 * size_fraction;
    float inh_spacing = 0.1 * inh_ratio *  size_fraction;

    // Add layers
    add_layer((new LayerConfig(name + "_pos",
        IZHIKEVICH, exc_rows, exc_cols))
            ->set_property(IZ_INIT, "random positive")
            ->set_property("spacing", std::to_string(exc_spacing)));

    add_layer((new LayerConfig(name + "_neg",
        IZHIKEVICH, inh_rows, inh_cols))
            ->set_property(IZ_INIT, "random negative")
            ->set_property("spacing", std::to_string(inh_spacing)));

    // Excitatory self connections
    int self_spread = 15;
    connect_layers(name + "_pos", name + "_pos",
        (new ConnectionConfig(
            true, 0, 0.5, CONVERGENT, ADD,
            //(new FlatWeightConfig(0.1, 0.1))
            (new LogNormalWeightConfig(-3.0, 1.0, 0.1))
                ->set_diagonal(false)))
        ->set_arborized_config(
            new ArborizedConfig(self_spread, 1, -self_spread/2))
        ->set_property("learning rate", learning_rate));

    // Exc -> Inh
    int exc_inh_spread = 15;
    connect_layers(name + "_pos", name + "_neg",
        (new ConnectionConfig(
            true, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.01, 1.0)))
        ->set_arborized_config(
            new ArborizedConfig(exc_inh_spread, 2, -exc_inh_spread/2))
        ->set_property("learning rate", learning_rate));

    // Inh -> Exc
    int inh_exc_spread = 5;
    connect_layers(name + "_neg", name + "_pos",
        (new ConnectionConfig(
            false, 0, 0.5, DIVERGENT, SUB,
            new FlatWeightConfig(0.1, 1.0)))
        ->set_arborized_config(
            new ArborizedConfig(inh_exc_spread, 2, -inh_exc_spread/2))
        ->set_property("learning rate", learning_rate));
}

void AuditoryCortex::connect_one_way(std::string name1, std::string name2,
        int spread, float fraction, int delay, int stride) {
    connect_layers(name1, name2,
        (new ConnectionConfig(
            true, delay, 0.5, CONVERGENT, ADD,
            //(new FlatWeightConfig(0.1, fraction))))
            (new LogNormalWeightConfig(-3.0, 0.5, 1.0))))
        ->set_arborized_config(
            new ArborizedConfig(spread, stride, -spread/2))
        ->set_property("learning rate", learning_rate));
}

void AuditoryCortex::add_input(std::string layer, std::string input_name,
        std::string module_name, std::string module_params) {
    add_layer(
        (new LayerConfig(input_name, IZHIKEVICH, 1, spec_size))
        ->set_property(IZ_INIT, "bursting"));
    add_module(input_name, module_name, module_params);

    int offset = cortex_rows - (spec_spread/2);
    for (int i = 0 ; i < spec_size; ++i) {
        connect_layers(
            input_name, layer,
            (new ConnectionConfig(false, 0, 1, SUBSET, ADD,
                new FlatWeightConfig(1.0, 1.0)))
            ->set_subset_config(
                new SubsetConfig(
                    0, 1,
                    i, i+1,
                    i * spec_spread, (i+1) * spec_spread,
                    0 + offset, spec_spread + offset))
            ->set_property("myelinated", "true"));
    }
}
