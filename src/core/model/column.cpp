#include "model/column.h"

#define IZHIKEVICH "izhikevich"
#define IZ_INIT "init"

Column::Column(std::string name, int cortex_size)
        : Structure(name, PARALLEL),
              cortex_size(cortex_size),
              exc_noise_mean(1.0),
              exc_noise_std_dev(0.1),
              exc_plastic(true),

              inh_ratio(2),
              inh_size(cortex_size/inh_ratio),
              inh_plastic(true),

              spread_ratio(4),
              spread_factor(spread_ratio * spread_ratio),
              spread(cortex_size / spread_ratio),
              intracortical_delay(2),

              thal_ratio(2),
              thal_size(cortex_size / thal_ratio),
              thal_spread(thal_size / spread_ratio),
              thal_noise_mean(0.0),
              thal_noise_std_dev(0.0),
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
    add_thalamocortical_reentry("6t", "6t");
}

void Column::connect(Column *col_a, Column *col_b,
        std::string name_a, std::string name_b) {
    static int intercortical_delay = 2;
    static float mean = 0.05;
    static float std_dev = 0.01;
    static float fraction = 1.0;
    static float max_weight = 1.0;
    static std::string conductance = "0.0025";
    static std::string learning_rate = "0.1";

    Structure::connect(
        col_a, name_a, col_b, name_b,
        (new ConnectionConfig(
            true, intercortical_delay, max_weight, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::add_neural_field(std::string field_name) {
    static std::string conductance = "0.0025";
    static std::string learning_rate = "0.1";

    std::string pos_name = field_name + "_" + "pos";
    std::string neg_name = field_name + "_" + "neg";

    add_layer((new LayerConfig(pos_name,
        IZHIKEVICH, cortex_size, cortex_size, exc_noise_mean, exc_noise_std_dev))
            ->set_property(IZ_INIT, "random positive"));
    add_layer((new LayerConfig(neg_name,
        IZHIKEVICH, inh_size, inh_size, 0.0, 0.0))
            ->set_property(IZ_INIT, "random negative"));

    // Excitatory -> Inhibitory Connection
    connect_layers(pos_name, neg_name,
        (new ConnectionConfig(false, 0, 4, FULLY_CONNECTED, ADD,
            new GaussianWeightConfig(1, 0.3, 0.1)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));

    // Inhibitory -> Excitatory Connection
    connect_layers(neg_name, pos_name,
        (new ConnectionConfig(
            inh_plastic, 0, inh_ratio*inh_ratio*4, FULLY_CONNECTED, SUB,
            new GaussianWeightConfig(
                0.05, 0.0, 1.0)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::connect_fields_one_way(std::string src, std::string dest) {
    static std::string conductance = "0.0025";
    static std::string learning_rate = "0.1";

    float max_weight = spread_factor;
    float fraction = 0.1;
    float mean = 1.0 * spread_factor / fraction;
    float std_dev = 0.3 * spread_factor / fraction;

    auto conn_type = (spread_factor == 1) ? FULLY_CONNECTED : CONVERGENT;

    connect_layers(
        src + "_pos", dest + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, conn_type, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction),
            new ArborizedConfig(spread,1,-spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::connect_fields_reentrant(std::string src, std::string dest) {
    static std::string conductance = "0.0025";
    static std::string learning_rate = "0.1";

    float max_weight = spread_factor;
    float mean = 0.05 * spread_factor;
    float std_dev = 0.01 * spread_factor;
    float fraction = 1.0;

    auto conn_type = (spread_factor == 1) ? FULLY_CONNECTED : CONVERGENT;

    connect_layers(
        src + "_pos", dest + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, conn_type, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction),
            new ArborizedConfig(spread,1,-spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
    connect_layers(
        dest + "_pos", src + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, conn_type, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction),
            new ArborizedConfig(spread,1,-spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::add_thalamic_nucleus() {
    static std::string conductance = "0.0025";
    static std::string learning_rate = "0.1";

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
        (new ConnectionConfig(false, 0, 4, FULLY_CONNECTED, ADD,
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
    static std::string conductance = "0.0025";
    static std::string learning_rate = "0.1";

    float base_mean = 1.0;
    float base_std_dev = 0.3;
    float fraction = 0.1;

    float in_max_weight = 1.0; //spread_factor;
    float in_mean = base_mean * spread_factor;
    float in_std_dev = base_std_dev * spread_factor;

    float out_max_weight = 1.0; //spread_factor * thal_ratio;
    float out_mean = base_mean * spread_factor * thal_ratio;
    float out_std_dev = base_std_dev * spread_factor * thal_ratio;

    auto in_conn_type = (spread_factor == 1) ? FULLY_CONNECTED : CONVERGENT;
    auto out_conn_type = (spread_factor == 1) ? FULLY_CONNECTED : DIVERGENT;

    connect_layers(
        src + "_pos", "thal_pos",
        (new ConnectionConfig(
            exc_plastic, thalamocortical_delay, in_max_weight, in_conn_type, ADD,
            new GaussianWeightConfig(in_mean, in_std_dev, fraction),
            new ArborizedConfig(thal_spread,thal_ratio,-thal_spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate)
        ->set_property("stp p", "0.7"));
    connect_layers(
        src + "_pos", "thal_neg",
        (new ConnectionConfig(
            false, thalamocortical_delay, in_max_weight, in_conn_type, ADD,
            new GaussianWeightConfig(in_mean, in_std_dev, fraction),
            new ArborizedConfig(thal_spread,thal_ratio,-thal_spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
    connect_layers(
        "thal_pos", dest + "_pos",
        (new ConnectionConfig(
            exc_plastic, thalamocortical_delay, out_max_weight, out_conn_type, ADD,
            new GaussianWeightConfig(out_mean, out_std_dev, fraction),
            new ArborizedConfig(spread,thal_ratio,-spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate)
        ->set_property("stp p", "0.7"));
}
