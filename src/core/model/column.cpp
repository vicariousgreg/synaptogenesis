#include "model/column.h"

#define IZHIKEVICH "izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

Column::Column(std::string name, int cortex_size)
        : Structure(name, PARALLEL),
              conductance(std::to_string(0.1 / (cortex_size * cortex_size))),
              learning_rate("0.01"),
              reentrant_learning_rate("0.01"),

              cortex_size(cortex_size),
              exc_noise_mean(0.15),
              exc_noise_std_dev(0.01),
              exc_plastic(false),

              inh_ratio(2),
              inh_size(cortex_size / inh_ratio),
              inh_noise_mean(0.15),
              inh_noise_std_dev(0.01),
              exc_inh_plastic(false),
              exc_inh_spread(3),
              inh_exc_spread(7),
              exc_inh_conductance(std::to_string(
                  0.5*inh_ratio / (exc_inh_spread * exc_inh_spread))),
              inh_exc_conductance(std::to_string(
                  0.5*inh_ratio / (inh_exc_spread * inh_exc_spread))),

              intracortical_delay(0),
              intercortical_delay(0),

              thal_ratio(2),
              thal_size(cortex_size / thal_ratio),
              thal_noise_mean(0.025),
              thal_noise_std_dev(0.01),
              thal_conductance(std::to_string(0.1 / (thal_size * thal_size))),
              thalamocortical_delay(10) {
    /*******************************************/
    /***************** CORTEX ******************/
    /*******************************************/
    //add_neural_field("2");
    add_neural_field("3a");
    add_neural_field("4");
    add_neural_field("56");
    add_neural_field("6t");

    /*******************************************/
    /************** INTRACORTICAL **************/
    /*******************************************/
    //connect_fields_one_way("2", "3a");
    connect_fields_one_way("4", "3a");
    connect_fields_one_way("4", "56");
    connect_fields_one_way("3a", "6t");

    connect_fields_reentrant("3a", "56");

    /*******************************************/
    /***************** THALAMUS ****************/
    /*******************************************/
    add_thalamic_nucleus();

    /*******************************************/
    /************ THALAMO-CORTICAL *************/
    /*******************************************/
    add_thalamocortical_reentry("6t", "3a");
}

void Column::add_input(int num_symbols,
        std::string module_name, std::string module_params) {
    // Input layer
    this->add_layer(new LayerConfig("input", RELAY, 1, num_symbols));
    this->add_module("input", module_name, module_params);

    // Input connection
    std::string input_conductance = "0.2";
    float fraction = 1.0;
    float mean = 0.1;
    float std_dev = 0.03;
    connect_layers("input", "4_pos",
        (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", input_conductance));
}

void Column::connect(Column *col_a, Column *col_b,
        std::string name_a, std::string name_b) {
    static float mean = 0.05;
    static float std_dev = 0.0;
    static float fraction = 1.0;
    static float max_weight = 1.0;

    Structure::connect(
        col_a, name_a, col_b, name_b,
        (new ConnectionConfig(
            col_b->exc_plastic, col_b->intercortical_delay,
            max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", col_b->conductance)
        ->set_property("learning rate", col_b->reentrant_learning_rate)
        ->set_property("stp p", "0.7"));
}

void Column::add_neural_field(std::string field_name) {
    std::string pos_name = field_name + "_" + "pos";
    std::string neg_name = field_name + "_" + "neg";

    add_layer((new LayerConfig(pos_name,
        IZHIKEVICH, cortex_size, cortex_size, exc_noise_mean, exc_noise_std_dev))
            //->set_property(IZ_INIT, "random positive"));
            ->set_property(IZ_INIT, "regular"));
    add_layer((new LayerConfig(neg_name,
        IZHIKEVICH, inh_size, inh_size, inh_noise_mean, inh_noise_std_dev))
            //->set_property(IZ_INIT, "random negative"));
            ->set_property(IZ_INIT, "fast"));

    // Excitatory -> Inhibitory Connection
    connect_layers(pos_name, neg_name,
        (new ConnectionConfig(
            exc_inh_plastic, 0, 4, CONVERGENT, ADD,
            new GaussianWeightConfig(0.1, 0.0, 1.0),
            new ArborizedConfig(exc_inh_spread, inh_ratio, -exc_inh_spread/2)))
        ->set_property("conductance", exc_inh_conductance)
        ->set_property("learning rate", learning_rate)
        ->set_property("stp p", "1.5"));

    // Inhibitory -> Excitatory Connection
    connect_layers(neg_name, pos_name,
        (new ConnectionConfig(
            false, 0, 4, DIVERGENT, SUB,
            new GaussianWeightConfig(0.1, 0.0, 1.0),
            new ArborizedConfig(inh_exc_spread, inh_ratio, -inh_exc_spread/2)))
        ->set_property("conductance", inh_exc_conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::connect_fields_one_way(std::string src, std::string dest) {
    float mean = 0.05;
    float std_dev = 0.0;
    float fraction = 1.0;
    float max_weight = 1.0;

    connect_layers(
        src + "_pos", dest + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::connect_fields_reentrant(std::string src, std::string dest) {
    float max_weight = 1.0;
    float mean = 0.05;
    float std_dev = 0.0;
    float fraction = 1.0;

    connect_layers(
        src + "_pos", dest + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", reentrant_learning_rate));
    connect_layers(
        dest + "_pos", src + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", reentrant_learning_rate));
}

void Column::add_thalamic_nucleus() {
    // Thalamic exc-inh pair
    std::string thal_pos_name = "thal_pos";
    std::string thal_neg_name = "thal_neg";
    add_layer((new LayerConfig(thal_pos_name,
        IZHIKEVICH, thal_size, thal_size, thal_noise_mean, thal_noise_std_dev))
            ->set_property(IZ_INIT, "thalamo_cortical"));
    add_layer((new LayerConfig(thal_neg_name,
        IZHIKEVICH, thal_size, thal_size, thal_noise_mean, thal_noise_std_dev))
            ->set_property(IZ_INIT, "thalamo_cortical"));

    // Excitatory -> Inhibitory Connection
    connect_layers(thal_pos_name, thal_neg_name,
        (new ConnectionConfig(
            exc_inh_plastic, 0, 4, CONVERGENT, ADD,
            new GaussianWeightConfig(0.1, 0.0, 1.0),
            new ArborizedConfig(exc_inh_spread, 1, -exc_inh_spread/2)))
        ->set_property("conductance", exc_inh_conductance)
        ->set_property("learning rate", learning_rate)
        ->set_property("stp p", "1.5"));

    // Inhibitory -> Excitatory Connection
    connect_layers(thal_neg_name, thal_pos_name,
        (new ConnectionConfig(
            false, 0, 4, DIVERGENT, SUB,
            new GaussianWeightConfig(0.1, 0.0, 1.0),
            new ArborizedConfig(inh_exc_spread, 1, -inh_exc_spread/2)))
        ->set_property("conductance", inh_exc_conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::add_thalamocortical_reentry(std::string src, std::string dest) {
    float max_weight = 1.0;
    float mean = 0.05;
    float std_dev = 0.0;
    float fraction = 1.0;

    connect_layers(
        src + "_pos", "thal_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", reentrant_learning_rate));
    connect_layers(
        "thal_pos", dest + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", thal_conductance)
        ->set_property("learning rate", reentrant_learning_rate));
}
