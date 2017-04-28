#include "model/column.h"

#define IZHIKEVICH "izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

/* Global shared variables */
static float base_conductance = 5;
static float noise_mean = 0.025;
static float noise_std_dev = 0.0;
static int intracortical_delay = 0;
static int intercortical_delay = 0;

// Thalamus variables
static int thal_ratio = 2;
static float thal_noise_mean = 0.025;
static float thal_noise_std_dev = 0.0025;
static int thalamocortical_delay = 10;

// Self connectivity variables
static float exc_self_base_conductance = 0.5;
static std::string exc_self_learning_rate = "0.01";
static std::string exc_self_stp_p = "0.7";
static std::string exc_self_stp_tau = "100";
static int exc_self_spread = 5;
static int exc_self_delay = 0;
static float exc_self_mean = -3.0;
static float exc_self_std_dev = 2.0;
static float exc_self_fraction = 1.0;

static float inh_self_base_conductance = 0.01;
static std::string inh_self_stp_p = "1.5";
static std::string inh_self_stp_tau = "100";
static int inh_self_spread = 9;
static int inh_self_delay = 0;
static float inh_self_mean = 1.0;     // Fixed, Gaussian
static float inh_self_std_dev = 0.0;  // Fixed, Gaussian
static float inh_self_fraction = 1.0; // Fixed, Gaussian

// Input variables
static std::string input_conductance = "0.2";
static std::string input_learning_rate = "0.01";
static float input_fraction = 1.0;
static float input_mean = -1.0;
static float input_std_dev = 0.5;

// One-way variables
static std::string one_way_learning_rate = "0.01";
static std::string one_way_stp_p = "1.0";
static std::string one_way_stp_tau = "100";
static float one_way_mean = -2.0;
static float one_way_std_dev = 1.0;
static float one_way_fraction = 1.0;
static float one_way_max_weight = 1.0;

// Reentrant variables
static std::string reentrant_learning_rate = "0.01";
static std::string reentrant_stp_p = "1.0";
static std::string reentrant_stp_tau = "100";
static float reentrant_mean = -2.0;
static float reentrant_std_dev = 1.0;
static float reentrant_fraction = 1.0;
static float reentrant_max_weight = 1.0;

Column::Column(std::string name, int cortex_size, bool plastic)
        : Structure(name, PARALLEL),
              conductance(std::to_string(
                  base_conductance
                  / (cortex_size * cortex_size))),

              cortex_size(cortex_size),
              exc_plastic(plastic),

              exc_inh_plastic(false),

              exc_self_conductance(std::to_string(
                  exc_self_base_conductance
                  / (exc_self_spread * exc_self_spread))),
              inh_self_conductance(std::to_string(
                  inh_self_base_conductance
                  / (inh_self_spread * inh_self_spread))),

              thal_size(cortex_size / thal_ratio),
              thal_conductance(std::to_string(0.1 / (thal_size * thal_size))) {
    /* Cortical Layers */
    //add_neural_field("2");
    add_neural_field("3a");
    add_neural_field("4");
    add_neural_field("56");
    //add_neural_field("6t");

    /* Intracortical Connections */
    // One-Way
    //connect_fields_one_way("2", "3a");
    connect_fields_one_way("4", "3a");
    connect_fields_one_way("4", "56");
    //connect_fields_one_way("56", "6t");

    // Reentrant
    //connect_fields_reentrant("3a", "56");

    /* Thalamic Nucleus */
    //add_thalamic_nucleus();
    //add_thalamocortical_reentry("6t", "6t");
}

void Column::add_input(bool plastic, int num_symbols,
        std::string module_name, std::string module_params) {
    // Input layer
    this->add_layer((new LayerConfig("input", IZHIKEVICH, 1, num_symbols))
        ->set_property(IZ_INIT, "regular"));
    this->add_module("input", module_name, module_params);

    // Input connection
    connect_layers("input", "4",
        (new ConnectionConfig(plastic, 0, 1, FULLY_CONNECTED, ADD,
            new LogNormalWeightConfig(
                input_mean, input_std_dev, input_fraction)))
        ->set_property("conductance", input_conductance)
        ->set_property("learning rate", input_learning_rate));
}

void Column::connect(Column *col_a, Column *col_b,
        std::string name_a, std::string name_b, bool reentrant) {
    std::string lr = (reentrant)
        ? one_way_learning_rate
        : reentrant_learning_rate;
    std::string stp_p = (reentrant)
        ? one_way_stp_p
        : reentrant_stp_p;
    std::string stp_tau = (reentrant)
        ? one_way_stp_tau
        : reentrant_stp_tau;
    float m =   (reentrant) ? one_way_mean       : reentrant_mean;
    float sd =  (reentrant) ? one_way_std_dev    : reentrant_std_dev;
    float max = (reentrant) ? one_way_max_weight : reentrant_max_weight;
    float f =   (reentrant) ? one_way_fraction   : reentrant_fraction;

    Structure::connect(
        col_a, name_a, col_b, name_b,
        (new ConnectionConfig(
            col_b->exc_plastic, intercortical_delay,
            max, FULLY_CONNECTED, ADD,
            new LogNormalWeightConfig(m, sd, f)))
        ->set_property("conductance", col_b->conductance)
        ->set_property("learning rate", lr)
        ->set_property("stp p", stp_p)
        ->set_property("stp tau", stp_tau));

    if (reentrant) {
        Structure::connect(
            col_a, name_a, col_b, name_b,
            (new ConnectionConfig(
                col_b->exc_plastic, intercortical_delay,
                max, FULLY_CONNECTED, ADD,
                new LogNormalWeightConfig(m, sd, f)))
            ->set_property("conductance", col_b->conductance)
            ->set_property("learning rate", lr)
            ->set_property("stp p", stp_p)
            ->set_property("stp tau", stp_tau));
    }
}

void Column::add_neural_field(std::string field_name) {
    add_layer((new LayerConfig(field_name,
        IZHIKEVICH, cortex_size, cortex_size, noise_mean, noise_std_dev))
            //->set_property(IZ_INIT, "random positive"));
            ->set_property(IZ_INIT, "regular"));

    // Excitatory self connections
    connect_layers(field_name, field_name,
        (new ConnectionConfig(
            exc_plastic, exc_self_delay, 1.0, CONVERGENT, ADD,
            new LogNormalWeightConfig(
                exc_self_mean, exc_self_std_dev, exc_self_fraction)))
        ->set_arborized_config(
            new ArborizedConfig(exc_self_spread, 1, -exc_self_spread/2))
        ->set_property("conductance", exc_self_conductance)
        ->set_property("learning rate", exc_self_learning_rate)
        ->set_property("stp p", exc_self_stp_p)
        ->set_property("stp tau", exc_self_stp_tau));

    // Inhibitory self connections
    connect_layers(field_name, field_name,
        (new ConnectionConfig(
            false, 0, 1.0, CONVERGENT, SUB,
            new GaussianWeightConfig(
                inh_self_mean, inh_self_std_dev, inh_self_fraction)))
        ->set_arborized_config(
            new ArborizedConfig(inh_self_spread, 1, -inh_self_spread/2))
        ->set_property("conductance", inh_self_conductance)
        ->set_property("stp p", inh_self_stp_p)
        ->set_property("stp tau", inh_self_stp_tau));
}

void Column::connect_fields_one_way(std::string src, std::string dest) {
    connect_layers(
        src, dest,
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, one_way_max_weight,
            FULLY_CONNECTED, ADD,
            new LogNormalWeightConfig(
                one_way_mean, one_way_std_dev, one_way_fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", one_way_learning_rate)
        ->set_property("stp p", one_way_stp_p)
        ->set_property("stp tau", one_way_stp_tau));
}

void Column::connect_fields_reentrant(std::string src, std::string dest) {
    connect_layers(
        src, dest,
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, reentrant_max_weight,
            FULLY_CONNECTED, ADD,
            new LogNormalWeightConfig(
                reentrant_mean, reentrant_std_dev, reentrant_fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", reentrant_learning_rate)
        ->set_property("stp p", reentrant_stp_p)
        ->set_property("stp tau", reentrant_stp_tau));

    connect_layers(
        dest, src,
        (new ConnectionConfig(
            exc_plastic, intracortical_delay, reentrant_max_weight,
            FULLY_CONNECTED, ADD,
            new LogNormalWeightConfig(
                reentrant_mean, reentrant_std_dev, reentrant_fraction)))
        ->set_property("conductance", conductance)
        ->set_property("learning rate", reentrant_learning_rate)
        ->set_property("stp p", reentrant_stp_p)
        ->set_property("stp tau", reentrant_stp_tau));
}

void Column::add_thalamic_nucleus() {
    /*
    // Thalamic exc-inh pair
    std::string thal_name = "thal";
    add_layer((new LayerConfig(thal_name,
        IZHIKEVICH, thal_size, thal_size, thal_noise_mean, thal_noise_std_dev))
            ->set_property(IZ_INIT, "thalamo_cortical"));
    */
}

void Column::add_thalamocortical_reentry(std::string src, std::string dest) {
}
