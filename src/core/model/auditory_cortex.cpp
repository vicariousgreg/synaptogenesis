#include "model/auditory_cortex.h"

#define LEAKY_IZHIKEVICH "leaky_izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

const std::string learning_rate = "1.0";
const int layer_spread = 31;
const int self_spread = 31;
const int inh_spread = 11;
const int spec_spacing = -3;

AuditoryCortex::AuditoryCortex(Model *model, int spec_size, int spec_spread)
        : Structure("Auditory Cortex", PARALLEL),
          spec_size(spec_size), spec_spread(spec_spread),
          cortex_rows((spec_spacing * (spec_size-1)) + (spec_spread * spec_size)),
          cortex_cols(3*spec_size) {
    this->add_cortical_layer("3b", false, 1);
    this->add_cortical_layer("5a", false, 2, 1);
    //this->add_cortical_layer("5b", false, 4, 2);

    //  DIFFERENCE BETWEEN MODELS HERE
    this->connect_one_way("3b_pos", "5a_pos", layer_spread, 0.1, 0, 2);
    //this->connect_reentrant("3b_pos", "5a_pos", layer_spread, 0.1, 3, 2);

    //this->connect_one_way("5a_pos", "5b_pos", layer_spread, 0.1, 0, 2);
    //this->connect_reentrant("3b_pos", "5a_pos", layer_spread, 0.1, 3, 2);
    //this->connect_reentrant("5a_pos", "5b_pos", 5, 0.1, 9, 2);

    model->add_structure(this);
}

void AuditoryCortex::add_cortical_layer(std::string name, bool shifted,
        int size_fraction, int conn_fraction) {
    int inh_ratio = 2;

    int exc_rows = cortex_rows / size_fraction;
    int exc_cols = cortex_cols / size_fraction;
    int inh_rows = exc_rows / inh_ratio;
    int inh_cols = exc_cols / inh_ratio;

    float exc_spacing = 0.1 * size_fraction;
    float inh_spacing = 0.1 * inh_ratio * size_fraction;

    // Add layers
    add_layer((new LayerConfig(name + "_pos",
        LEAKY_IZHIKEVICH, exc_rows, exc_cols))
            ->set_property(IZ_INIT, "random positive")
            ->set_property("spacing", std::to_string(exc_spacing)));

    add_layer((new LayerConfig(name + "_neg",
        LEAKY_IZHIKEVICH, inh_rows, inh_cols))
            ->set_property(IZ_INIT, "random negative")
            ->set_property("spacing", std::to_string(inh_spacing)));

    // Excitatory self connections
    int spread = self_spread / conn_fraction;
    connect_layers(name + "_pos", name + "_pos",
        (new ConnectionConfig(
            true, 0, 0.5, CONVERGENT, ADD,
            new SurroundWeightConfig(1,1, new FlatWeightConfig(0.05, 0.1))))
            //(new LogNormalWeightConfig(-3.0, 1.0, 0.1))
        ->set_arborized_config(
            new ArborizedConfig(
                spread,
                spread,
                1,
                1,
                -spread/2,
                (shifted) ? (-spread - spec_spread) : (-spread/2)))
        ->set_property("short term plasticity", (shifted) ? "false" : "true")
        ->set_property("learning rate", learning_rate));

    // Exc -> Inh
    int exc_inh_spread = self_spread / conn_fraction;
    connect_layers(name + "_pos", name + "_neg",
        (new ConnectionConfig(
            false, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.05, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(
                exc_inh_spread,
                exc_inh_spread,
                2,
                2,
                -exc_inh_spread/2,
                -exc_inh_spread/2))
                //(shifted) ? (-exc_inh_spread - spec_spread) : (-exc_inh_spread/2)))
        ->set_property("myelinated", "true")
        ->set_property("learning rate", learning_rate));

    // Inh -> Exc
    int inh_exc_spread = inh_spread / conn_fraction;
    connect_layers(name + "_neg", name + "_pos",
        (new ConnectionConfig(
            false, 0, 0.5, DIVERGENT, SUB,
            new FlatWeightConfig(0.05, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(
                inh_exc_spread,
                inh_exc_spread,
                2,
                2,
                -inh_exc_spread/2,
                -inh_exc_spread/2))
                //(shifted) ? (-inh_exc_spread - spec_spread) : (-inh_exc_spread/2)))
        ->set_property("learning rate", learning_rate));
}

void AuditoryCortex::connect_one_way(std::string name1, std::string name2,
        int spread, float fraction, int delay, int ratio) {
    connect_layers(name1, name2,
        (new ConnectionConfig(
            true, delay, 0.5, CONVERGENT, ADD,
            (new FlatWeightConfig(0.1, fraction))))
            //(new LogNormalWeightConfig(-3.0, 0.5, 1.0))))
        ->set_arborized_config(
            new ArborizedConfig(spread, ratio, -spread/2))
        //->set_property("myelinated", "true")
        ->set_property("learning rate", learning_rate));
}

void AuditoryCortex::connect_reentrant(std::string name1, std::string name2,
        int spread, float fraction, int delay, int ratio) {
    connect_one_way(name1, name2, spread, fraction,delay, ratio);

    int feedback_spread = spread / ratio;
    connect_layers(name2, name1,
        (new ConnectionConfig(
            true, delay, 0.5, DIVERGENT, ADD,
            (new FlatWeightConfig(0.1, fraction))))
            //(new LogNormalWeightConfig(-3.0, 0.5, 1.0))))
        ->set_arborized_config(
            new ArborizedConfig(feedback_spread, ratio, -feedback_spread/2))
        //->set_property("myelinated", "true")
        ->set_property("learning rate", learning_rate));
}

void AuditoryCortex::add_input(std::string layer, std::string input_name,
        ModuleConfig *config) {
    add_layer(
        (new LayerConfig(input_name, LEAKY_IZHIKEVICH, 1, spec_size))
        ->set_property(IZ_INIT, "regular"));
    add_module(input_name, config);

    int size = spec_spacing + spec_spread;
    int offset = (cortex_cols / 2) - (size / 2);
    for (int i = 0 ; i < spec_size; ++i) {
        connect_layers(
            input_name, layer,
            (new ConnectionConfig(false, 0, 1, SUBSET, ADD,
                //new FlatWeightConfig(1.0, 1.0)))
                new GaussianWeightConfig(0.1, 0.25, 0.01)))
            ->set_subset_config(
                new SubsetConfig(
                    0, 1,
                    i, i+1,
                    i * size,
                    i * size + spec_spread,
                    //offset + 0, offset + spec_spread))
                    0, cortex_cols))
            ->set_property("myelinated", "true")
            ->set_property("short term plasticity", "false"));
    }
}
