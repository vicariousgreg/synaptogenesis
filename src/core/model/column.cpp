#include "model/column.h"

#define IZHIKEVICH "izhikevich"
#define IZ_INIT "init"

Column::Column(std::string name, int cortex_size)
        : Structure(name, PARALLEL),
              conductance(std::to_string(10.0 / (cortex_size * cortex_size))),
              learning_rate("0.1"),

              cortex_size(cortex_size),
              exc_noise_mean(0.1),
              exc_noise_std_dev(0.01),
              exc_plastic(true),

              inh_ratio(2),
              inh_size(cortex_size / inh_ratio),
              inh_noise_mean(0.1),
              inh_noise_std_dev(0.01),
              exc_inh_plastic(false),
              inh_plastic(false),
              exc_inh_spread(3),
              inh_exc_spread(7),
              exc_inh_conductance(std::to_string(
                  0.05*inh_ratio / (exc_inh_spread * exc_inh_spread))),
              inh_exc_conductance(std::to_string(
                  0.05*inh_ratio / (inh_exc_spread * inh_exc_spread))),

              intracortical_delay(2),

              thal_ratio(2),
              thal_size(cortex_size / thal_ratio),
              thal_noise_mean(0.0),
              thal_noise_std_dev(0.0),
              thalamocortical_delay(10) {
    /*******************************************/
    /***************** CORTEX ******************/
    /*******************************************/
    //add_neural_field("2");
    add_neural_field("3a");
    add_neural_field("4");
    //add_neural_field("56");
    //add_neural_field("6t");

    /*******************************************/
    /************** INTRACORTICAL **************/
    /*******************************************/
    //connect_fields_one_way("2", "3a");
    connect_fields_one_way("4", "3a");
    //connect_fields_one_way("4", "56");
    //connect_fields_one_way("3a", "6t");

    //connect_fields_reentrant("3a", "56");

    /*******************************************/
    /***************** THALAMUS ****************/
    /*******************************************/
    //add_thalamic_nucleus();

    /*******************************************/
    /************ THALAMO-CORTICAL *************/
    /*******************************************/
    //add_thalamocortical_reentry("6t", "6t");
}

void Column::connect(Column *col_a, Column *col_b,
        std::string name_a, std::string name_b) {
    static int intercortical_delay = 2;
    static float mean = 0.05;
    static float std_dev = 0.01;
    static float fraction = 1.0;
    static float max_weight = 4.0;

    Structure::connect(
        col_a, name_a, col_b, name_b,
        (new ConnectionConfig(
            true, intercortical_delay, max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", col_b->conductance)
        ->set_property("learning rate", col_b->learning_rate)
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
            new GaussianWeightConfig(1.0, 0.0, 1.0),
            new ArborizedConfig(exc_inh_spread, inh_ratio, -exc_inh_spread/2)))
        ->set_property("conductance", exc_inh_conductance)
        ->set_property("learning rate", learning_rate)
        ->set_property("stp p", "1.5"));

    // Inhibitory -> Excitatory Connection
    connect_layers(neg_name, pos_name,
        (new ConnectionConfig(
            inh_plastic, 0, 4, DIVERGENT, SUB,
            new GaussianWeightConfig(1.0, 0.0, 1.0),
            new ArborizedConfig(inh_exc_spread, inh_ratio, -inh_exc_spread/2)))
        ->set_property("conductance", inh_exc_conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::connect_fields_one_way(std::string src, std::string dest) {
    float mean = 1.0;
    float std_dev = 0.5;
    float fraction = 1.0;
    float max_weight = 4.0;

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
    float std_dev = 0.01;
    float fraction = 1.0;

    connect_layers(
        src + "_pos", dest + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
    connect_layers(
        dest + "_pos", src + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::add_thalamic_nucleus() {
    // Thalamic exc-inh pair
    std::string thal_pos_name = "thal_pos";
    std::string thal_neg_name = "thal_neg";
    add_layer((new LayerConfig(thal_pos_name,
        IZHIKEVICH, thal_size, thal_size, thal_noise_mean, thal_noise_std_dev))
            ->set_property(IZ_INIT, "thalamo_cortical"));
    add_layer((new LayerConfig(thal_neg_name,
        IZHIKEVICH, thal_size, thal_size, 0.0, 0.0))
            ->set_property(IZ_INIT, "thalamo_cortical"));

    // Excitatory -> Inhibitory Connection
    connect_layers(thal_pos_name, thal_neg_name,
        (new ConnectionConfig(exc_inh_plastic, 0, 4, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(1, 0.3, 0.1)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate)
        ->set_property("stp p", "1.5"));

    // Inhibitory -> Excitatory Connection
    connect_layers(thal_neg_name, thal_pos_name,
        (new ConnectionConfig(inh_plastic, 0, 4, FULLY_CONNECTED, SUB,
            new GaussianWeightConfig(0.05, 0.0, 1.0)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::add_thalamocortical_reentry(std::string src, std::string dest) {
    float base_mean = 1.0;
    float base_std_dev = 0.3;
    float fraction = 0.1;

    float in_max_weight = 1.0;
    float in_mean = base_mean;
    float in_std_dev = base_std_dev;

    float out_max_weight = 1.0;
    float out_mean = base_mean * thal_ratio;
    float out_std_dev = base_std_dev * thal_ratio;

    connect_layers(
        src + "_pos", "thal_pos",
        (new ConnectionConfig(
            exc_plastic, thalamocortical_delay, in_max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(in_mean, in_std_dev, fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate)
        ->set_property("stp p", "0.7"));
    connect_layers(
        src + "_pos", "thal_neg",
        (new ConnectionConfig(
            false, thalamocortical_delay, in_max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(in_mean, in_std_dev, fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
    connect_layers(
        "thal_pos", dest + "_pos",
        (new ConnectionConfig(
            exc_plastic, thalamocortical_delay, out_max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(out_mean, out_std_dev, fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate)
        ->set_property("stp p", "0.7"));
}
