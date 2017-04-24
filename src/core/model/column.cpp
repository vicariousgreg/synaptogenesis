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
    add_thalamocortical_reentry("6t");
}

void Column::connect(Column *col_a, Column *col_b,
        std::string name_a, std::string name_b) {
    static int intercortical_delay = 2;
    static float mean = 0.0;
    static float std_dev = 0.0;
    static float fraction = 1.0;
    static float max_weight = 1.0;
    static std::string conductance = "0.01";
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
    static std::string conductance = "0.01";
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
                inh_ratio*inh_ratio*1, inh_ratio*inh_ratio*0.3, 0.1)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::connect_fields_one_way(std::string src, std::string dest) {
    static std::string conductance = "0.01";
    static std::string learning_rate = "0.1";

    float max_weight = spread_ratio;
    float fraction = 0.05;
    float mean = 1.0 * spread_ratio / fraction;
    float std_dev = 0.3 * spread_ratio / fraction;

    connect_layers(
        src + "_pos", dest + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, CONVERGENT, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction),
            new ArborizedConfig(spread,1,-spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::connect_fields_reentrant(std::string src, std::string dest) {
    static std::string conductance = "0.01";
    static std::string learning_rate = "0.1";

    float max_weight = spread_ratio;
    float mean = 0.05 * spread_ratio;
    float std_dev = 0.01 * spread_ratio;
    float fraction = 1.0;

    connect_layers(
        src + "_pos", dest + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, CONVERGENT, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction),
            new ArborizedConfig(spread,1,-spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
    connect_layers(
        dest + "_pos", src + "_pos",
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, max_weight, CONVERGENT, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction),
            new ArborizedConfig(spread,1,-spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::add_thalamic_nucleus() {
    static std::string conductance = "0.01";
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
        ->set_property("learning rate", learning_rate));

    // Inhibitory -> Excitatory Connection
    connect_layers(thal_neg_name, thal_pos_name,
        (new ConnectionConfig(inh_plastic, 0, 4, FULLY_CONNECTED, SUB,
            new GaussianWeightConfig(1, 0.3, 0.1)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}

void Column::add_thalamocortical_reentry(std::string src) {
    static std::string conductance = "0.01";
    static std::string learning_rate = "0.1";

    float max_weight = spread_ratio * thal_ratio;
    float mean = 0.05 * spread_ratio * thal_ratio;
    float std_dev = 0.01 * spread_ratio * thal_ratio;
    float fraction = 1.0;

    std::string dest = "thal_pos";
    connect_layers(
        src + "_pos", dest,
        (new ConnectionConfig(
            exc_plastic, thalamocortical_delay, max_weight, CONVERGENT, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction),
            new ArborizedConfig(thal_spread,thal_ratio,-thal_spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
    connect_layers(
        dest, src + "_pos",
        (new ConnectionConfig(
            exc_plastic, thalamocortical_delay, max_weight, DIVERGENT, ADD,
            new GaussianWeightConfig(mean, std_dev, fraction),
            new ArborizedConfig(thal_spread,thal_ratio,-thal_spread/2)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", learning_rate));
}
