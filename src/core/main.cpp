#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "network/network.h"
#include "builder.h"
#include "network/weight_config.h"
#include "io/module.h"
#include "io/impl/dsst_module.h"
#include "io/environment.h"
#include "engine/engine.h"
#include "state/state.h"
#include "util/tools.h"
#include "engine/context.h"

#define IZHIKEVICH "izhikevich"
#define HEBBIAN_RATE_ENCODING "hebbian_rate_encoding"
#define BACKPROP_RATE_ENCODING "backprop_rate_encoding"
#define RELAY "relay"

#define IZ_INIT "init"

void print_network(Network *network, Environment *env=nullptr) {
    printf("Built network.\n");
    printf("  - neurons     : %10d\n", network->get_num_neurons());
    printf("  - layers      : %10d\n", network->get_num_layers());
    printf("  - connections : %10d\n", network->get_num_connections());
    printf("  - weights     : %10d\n", network->get_num_weights());

    for (auto structure : network->get_structures()) {
        for (auto layer : structure->get_layers()) {
            printf("%-20s   ", (layer->structure->name + "->" + layer->name).c_str());
            /*
            if (env != nullptr) {
                auto io_type = env->get_io_type(structure->name, layer->name);
                std::cout << ((io_type & INPUT) ? "I " : "  ");
                std::cout << ((io_type & OUTPUT) ? "O " : "  ");
                std::cout << ((io_type & EXPECTED) ? "E " : "  ");
            }
            */
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void old_test() {
    /* Construct the network */
    Network *network = new Network();
    Structure *structure = new Structure("old");
    network->add_structure(structure);

    std::string model_name = "leaky_izhikevich";

    int resolution = 128;
    // structure->add_layer((new LayerConfig(
    //     "input_layer", model_name, 1, 10))
    //        ->set(IZ_INIT, "default"));
    structure->add_layer(
        (new LayerConfig(
            "exc_thalamus", model_name, resolution, resolution,
            (new PropertyConfig())
            ->set_value("type", "poisson")))
        ->set(IZ_INIT, "thalamo_cortical"));
    // structure->add_layer((new LayerConfig(
    //     "inh_thalamus", model_name, resolution/2, resolution/2))
    //         ->set("spacing", "0.2")
    //         ->set(IZ_INIT, "random negative"));
    structure->add_layer((new LayerConfig(
        "exc_cortex", model_name, resolution, resolution))
            ->set(IZ_INIT, "random positive"));
    structure->add_layer((new LayerConfig(
        "inh_cortex", model_name, resolution/2, resolution/2))
            ->set("spacing", "0.2")
            ->set(IZ_INIT, "random negative"));

    /* Forward excitatory pathway */
    // structure->connect_layers("input_layer", "exc_thalamus",
    //     (new ConnectionConfig(false, 0, 5, FULLY_CONNECTED, ADD,
    //         new RandomWeightConfig(1, 0.01)))
    //     ->set("myelinated", "true"));
    structure->connect_layers("exc_thalamus", "exc_cortex",
        (new ConnectionConfig(true, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(15,1,true)));
    structure->connect_layers("exc_cortex", "exc_cortex",
        (new ConnectionConfig(true, 2, 5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(31,1,true)));

    /* Cortical inhibitory loop */
    structure->connect_layers("exc_cortex", "inh_cortex",
        (new ConnectionConfig(false, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(31,2,true)));
    structure->connect_layers("inh_cortex", "exc_cortex",
        (new ConnectionConfig(false, 0, 1, DIVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(5,2,true)));

    /* Cortico-thalamic inhibitory loop */
    /*
    structure->connect_layers("exc_cortex", "inh_thalamus",
        (new ConnectionConfig(true, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(7,2,true)));
    structure->connect_layers("inh_thalamus", "exc_thalamus",
        (new ConnectionConfig(false, 0, 1, DIVERGENT, SUB,
            new FlatWeightConfig(0.5, 0.1)))
        ->set("myelinated", "false")
        ->set_arborized_config(new ArborizedConfig(5,2,true)));
    */


    // Modules
    auto env = new Environment();

    env->add_module(
        (new ModuleConfig("visualizer"))
        //->add_layer("old", "inh_thalamus")
        //->add_layer("old", "inh_cortex")
        ->add_layer("old", "exc_thalamus")
        ->add_layer("old", "exc_cortex"));

    env->add_module(
        (new ModuleConfig("heatmap"))
        //->add_layer("old", "inh_thalamus")
        //->add_layer("old", "inh_cortex")
        ->add_layer("old", "exc_thalamus")
        ->add_layer("old", "exc_cortex"));

    // env->add_module(
    //     (new ModuleConfig("periodic_input", "old", "input_layer"))
    //     ->set("max", "5")
    //     ->set("random", "true")
    //     ->set("rate", "1000000")
    //     ->set("verbose", "true"));

    print_network(network, env);
    Engine engine(new Context(network, env));
    delete engine.run(1000000, true);
}

void simple_test() {
    /* Construct the network */
    Network *network = new Network();
    Structure *structure = new Structure("simple");
    network->add_structure(structure);

    std::string model_name = "leaky_izhikevich";

    int exc_field = 25;
    int inh_field = 15;

    int resolution = 96;
    structure->add_layer((new LayerConfig(
        "input_layer", model_name, 1, 10))
            ->set(IZ_INIT, "regular"));
    structure->add_layer(
        (new LayerConfig(
            "hid_1", model_name, resolution, resolution,
            (new PropertyConfig())
                ->set_value("type", "poisson")
                ->set_value("value", "20")
                ->set_value("rate", "1")))
            ->set(IZ_INIT, "regular"));
    structure->add_layer((new LayerConfig(
        "hid_2", model_name, resolution, resolution))
            ->set(IZ_INIT, "regular"));

    /* Forward excitatory pathway */
    /*
    structure->connect_layers("input_layer", "hid_1",
        (new ConnectionConfig(false, 0, 0.5, FULLY_CONNECTED, ADD,
            new RandomWeightConfig(1, 0.05)))
        ->set("myelinated", "true"));
    */

    structure->connect_layers("hid_1", "hid_2",
        (new ConnectionConfig(true, 10, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field,1,true)));
    /*
    structure->connect_layers("hid_1", "hid_2",
        (new ConnectionConfig(false, 10, 0.5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field,1,true)));
    */

    /* Recurrent self connectivity */
    structure->connect_layers("hid_1", "hid_1",
        (new ConnectionConfig(true, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field,1,true)));
    structure->connect_layers("hid_1", "hid_1",
        (new ConnectionConfig(false, 0, 0.5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field,1,true)));

    structure->connect_layers("hid_2", "hid_2",
        (new ConnectionConfig(true, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field,1,true)));
    structure->connect_layers("hid_2", "hid_2",
        (new ConnectionConfig(false, 0, 0.5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field,1,true)));

    /* Feedback connectivity */
    structure->connect_layers("hid_2", "hid_1",
        (new ConnectionConfig(true, 10, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(exc_field,1,true)));
    /*
    structure->connect_layers("hid_2", "hid_1",
        (new ConnectionConfig(false, 10, 0.5, CONVERGENT, SUB,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(
            new ArborizedConfig(inh_field,1,true)));
    */

    // Modules
    auto env = new Environment();
    env->add_module(
        (new ModuleConfig("one_hot_random_input", "simple", "input_layer"))
            ->set("max", "4")
            ->set("rate", "1000000"));

    env->add_module(
        (new ModuleConfig("visualizer"))
        ->add_layer("simple", "input_layer")
        ->add_layer("simple", "hid_1")
        ->add_layer("simple", "hid_2"));
    env->add_module(
        (new ModuleConfig("heatmap"))
        ->add_layer("simple", "input_layer")
        ->add_layer("simple", "hid_1")
        ->add_layer("simple", "hid_2"));

    auto c = new Context(network, env);

    std::string filename = "simple.bin";

    if (State::exists(filename)) {
        print_network(network, env);
        c->get_state()->load("simple.bin");
        Engine engine(c, true);
        delete engine.run(1000000, true);
    } else {
        print_network(network, env);
        Engine engine(c, true);
        engine.set_learning_flag(true);
        engine.run(500000, true);
        c->get_state()->save("simple.bin");
        delete c;
    }
}

void single_field_test() {
    /* Construct the network */
    Network *network = new Network();
    Structure *structure = new Structure("single field");
    network->add_structure(structure);

    std::string model_name = "leaky_izhikevich";

    structure->add_layer((new LayerConfig(
        "exc_field", model_name, 40, 40))
            ->set(IZ_INIT, "random positive"));
    structure->add_layer((new LayerConfig(
        "inh_field", model_name, 10, 40))
            ->set(IZ_INIT, "random negative"));

    structure->connect_layers("exc_field", "exc_field",
        (new ConnectionConfig(true, 0, 10, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(.6, 0.1)))
        ->set("random delay", "20"));
    structure->connect_layers("exc_field", "inh_field",
        (new ConnectionConfig(true, 0, 10, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(.6, 0.1)))
        ->set("random delay", "20"));

    structure->connect_layers("inh_field", "inh_field",
        (new ConnectionConfig(false, 0, 10, FULLY_CONNECTED, SUB,
            new FlatWeightConfig(.5, 0.1)))
        ->set("myelinated", "true"));
    structure->connect_layers("inh_field", "exc_field",
        (new ConnectionConfig(false, 0, 10, FULLY_CONNECTED, SUB,
            new FlatWeightConfig(.5, 0.1)))
        ->set("myelinated", "true"));

    // Modules
    auto env = new Environment();
    env->add_module(
        (new ModuleConfig("one_hot_random_input", "single field", "exc_field"))
            ->set("max", "20")
            ->set("rate", "1")
            ->set("verbose", "false"));
    env->add_module(
        (new ModuleConfig("visualizer"))
        ->add_layer("single field", "exc_field")
        ->add_layer("single field", "inh_field"));
    env->add_module(
        (new ModuleConfig("heatmap"))
        ->add_layer("single field", "exc_field")
        ->add_layer("single field", "inh_field"));

    auto c = new Context(network, env);

    print_network(network, env);
    Engine engine(c);
    delete engine.run(1000000, true);
}

void mnist_test() {
    /* Construct the network */
    Network *network = new Network();
    Structure *structure = new Structure("mnist");
    network->add_structure(structure);

    structure->add_layer((new LayerConfig("input_layer",
        IZHIKEVICH, 28, 28))
            ->set(IZ_INIT, "regular"));

    // Hidden distributed layer
    structure->add_layer(
        (new LayerConfig("hidden",
            "leaky_izhikevich", 28*3, 28*3))
            ->set(IZ_INIT, "regular"));
    structure->connect_layers("input_layer", "hidden",
        (new ConnectionConfig(true, 0, 0.5, DIVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(new ArborizedConfig(21,3,false))
        ->set("myelinated", "false"));
    /*
    structure->connect_layers("hidden", "input_layer",
        (new ConnectionConfig(true, 0, 0.5, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(new ArborizedConfig(9,1,true))
        ->set("myelinated", "true"));
    */

    structure->connect_layers("hidden", "hidden",
        (new ConnectionConfig(true, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(new ArborizedConfig(25,1,false)));
    structure->connect_layers("hidden", "hidden",
        (new ConnectionConfig(false, 0, 1, CONVERGENT, SUB,
            new SurroundWeightConfig(25, 25, new FlatWeightConfig(0.1, 0.1))))
        ->set_arborized_config(new ArborizedConfig(31,1,false)));

    // Topological separation layer
    /* ////////
    structure->add_layer(
        (new LayerConfig("separation",
            "leaky_izhikevich", 10, 100))
            ->set(IZ_INIT, "regular"));
    structure->connect_layers("hidden", "separation",
        (new ConnectionConfig(true, 10, 0.5, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(0.01, 0.1)))
        ->set_arborized_config(new ArborizedConfig(9,1,true))
        ->set("myelinated", "true"));
    structure->connect_layers("separation", "hidden",
        (new ConnectionConfig(true, 10, 0.5, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(0.01, 0.1)))
        ->set_arborized_config(new ArborizedConfig(9,1,true))
        ->set("myelinated", "true"));

    / *
    structure->connect_layers("separation", "separation",
        (new ConnectionConfig(true, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1, 0.1)))
        ->set_arborized_config(new ArborizedConfig(9,1,true)));
    * /
    structure->connect_layers("separation", "separation",
        (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, SUB,
            new FlatWeightConfig(0.01, 0.1)))
        ->set_arborized_config(new ArborizedConfig(11,1,true))
        ->set("myelinated", "true"));
    */ /////////////

    // Discrete output layer
    structure->add_layer((new LayerConfig("output_layer",
        IZHIKEVICH, 1, 10))
            ->set(IZ_INIT, "regular"));

    /* //////////
    for (int i = 0 ; i < 10 ; ++i) {
        structure->connect_layers("output_layer", "separation",
            (new ConnectionConfig(false, 0, 0.5, SUBSET, ADD,
                new RandomWeightConfig(2)))
            ->set("myelinated", "true")
            ->set_subset_config(
                new SubsetConfig(
                    0,1,
                    i,i+1,
                    0,10,
                    10*i, 10*(i+1))));
    }
    */ ///////////

    // Modules
    std::string input_file = "/HDD/datasets/mnist/processed/mnist_test_input.csv";
    std::string output_file = "/HDD/datasets/mnist/processed/mnist_test_output.csv";
    auto env = new Environment();
    env->add_module(
        (new ModuleConfig("csv_input", "mnist", "input_layer"))
        ->set("filename", input_file)
        ->set("offset", "0")
        ->set("exposure", "5000")
        ->set("normalization", "25"));
    env->add_module(
        (new ModuleConfig("csv_input", "mnist", "output_layer"))
        ->set("filename", output_file)
        ->set("offset", "0")
        ->set("exposure", "5000")
        ->set("normalization", "0.2"));

    env->add_module(
        (new ModuleConfig("visualizer"))
        ->add_layer("mnist", "input_layer")
        ->add_layer("mnist", "hidden")
        //->add_layer("mnist", "separation")
        ->add_layer("mnist", "output_layer"));
    env->add_module(
        (new ModuleConfig("heatmap"))
        ->add_layer("mnist", "input_layer")
        ->add_layer("mnist", "hidden")
        //->add_layer("mnist", "separation")
        ->add_layer("mnist", "output_layer"));

    std::cout << "MNIST test......\n";
    print_network(network, env);

    auto c = new Context(network, env);
    Engine engine(c);
    engine.run(1000000, true);
    c->get_state()->save("mnist.bin");

    delete c;
    std::cout << "\n";
}

void mnist_perceptron_test() {
    /* Construct the network */
    Network *network = new Network();
    Structure *structure = new Structure("mnist", FEEDFORWARD);
    network->add_structure(structure);

    structure->add_layer(new LayerConfig("input_layer", "relay", 28, 28));
    structure->add_layer(new LayerConfig("output_layer", "perceptron", 1, 10));
    structure->add_layer(new LayerConfig("bias_layer", "relay", 1, 1));

    structure->connect_layers("input_layer", "output_layer",
        new ConnectionConfig(true, 0, 1, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(0)));
    structure->connect_layers("bias_layer", "output_layer",
        new ConnectionConfig(true, 0, 1, FULLY_CONNECTED, ADD,
            new FlatWeightConfig(0)));

    // Modules for training
    std::string input_file = "/HDD/datasets/mnist/processed/mnist_train_input.csv";
    std::string output_file = "/HDD/datasets/mnist/processed/mnist_train_output.csv";
    auto env = new Environment();
    env->add_module(
        (new ModuleConfig("csv_input", "mnist", "input_layer"))
        ->set("filename", input_file)
        ->set("offset", "0")
        ->set("exposure", "1")
        ->set("normalization", "255"));
    env->add_module(
        (new ModuleConfig("csv_evaluator", "mnist", "output_layer"))
        ->set("filename", output_file)
        ->set("offset", "0")
        ->set("exposure", "1")
        ->set("normalization", "1"));
    env->add_module(
        new ModuleConfig("periodic_input", "mnist", "bias_layer"));

    // Run training
    auto c = new Context(network, env);
    Engine engine(c);
    engine.set_calc_rate(false);
    engine.run(60000, true);

    // Remove modules and replace for testing
    env->remove_modules();

    input_file = "/HDD/datasets/mnist/processed/mnist_test_input.csv";
    output_file = "/HDD/datasets/mnist/processed/mnist_test_output.csv";
    env->add_module(
        (new ModuleConfig("csv_input", "mnist", "input_layer"))
        ->set("filename", input_file)
        ->set("offset", "0")
        ->set("exposure", "1")
        ->set("normalization", "255"));
    env->add_module(
        (new ModuleConfig("csv_evaluator", "mnist", "output_layer"))
        ->set("filename", output_file)
        ->set("offset", "0")
        ->set("exposure", "1")
        ->set("normalization", "1"));
    env->add_module(
        new ModuleConfig("periodic_input", "mnist", "bias_layer"));

    // Run testing (disable learning)
    engine.rebuild();
    engine.set_learning_flag(false);
    engine.run(10000, true);

    delete c;
}

void game_of_life_test() {
    /* Construct the network */
    Network *network = new Network();
    Structure *structure = new Structure("game_of_life");
    network->add_structure(structure);
    std::string model_name = "game_of_life";

    // Contour rules
    // R4,C0,M1,S41..81, B41..81,NM
    // R7,C0,M1,S113..225, B113..225,NM

    // Game parameters
    bool wrap = true;
    int board_dim = 256;
    int neighborhood_size = 5; 15;
    int survival_min = 2; 113;
    int survival_max = 3; 225;
    int birth_min = 3; 113;
    int birth_max = 4; 225;

    // Input parameters
    bool one_step = false;
    float one_step_fraction = 0.5;
    float random_fraction = 0.5;
    float rate = 1000;

    structure->add_layer((new LayerConfig(
        "board", model_name, board_dim, board_dim,
        (new PropertyConfig())
        ->set_value("type", "poisson")
        ->set_value("value", std::to_string(birth_min))
        ->set_value("rate", "0.5")))
            ->set("survival_min", std::to_string(survival_min))
            ->set("survival_max", std::to_string(survival_max))
            ->set("birth_min", std::to_string(birth_min))
            ->set("birth_max", std::to_string(birth_max)));
    structure->connect_layers("board", "board",
        (new ConnectionConfig(false, 0, 1.0, CONVOLUTIONAL, ADD,
            new SurroundWeightConfig(1,1, new FlatWeightConfig(1))))
        ->set_arborized_config(
            new ArborizedConfig(neighborhood_size, 1, -neighborhood_size/2, wrap)));
    structure->connect_layers("board", "board",
        (new ConnectionConfig(false, 0, 1.0, CONVOLUTIONAL, SUB,
            new SurroundWeightConfig(neighborhood_size,neighborhood_size,
                new FlatWeightConfig(1))))
        ->set_arborized_config(
            new ArborizedConfig(neighborhood_size+2, 1, (-neighborhood_size+1)/2, wrap)));

    // Modules
    auto env = new Environment();

    /*
    if (one_step)
        // Single Initial State
        // Set the end to timestep 1
        env->add_module(
            (new ModuleConfig("periodic_input", "game_of_life", "board"))
                ->set("end", "1")
                ->set("max", std::to_string(birth_min))
                ->set("random", "false")
                ->set("verbose", "false")
                ->set("fraction", std::to_string(one_step_fraction)));
    else
        // Refresh state
        env->add_module(
            (new ModuleConfig("periodic_input", "game_of_life", "board"))
                ->set("max", std::to_string(birth_min))
                ->set("rate", std::to_string(rate))
                ->set("clear", "true")
                ->set("verbose", "false")
                ->set("fraction", std::to_string(random_fraction)));
    */

    env->add_module(
        new ModuleConfig("visualizer", "game_of_life", "board"));

    auto c = new Context(network, env);
    Engine engine(c);
    print_network(network, env);
    delete engine.run(500000, true);
}

void working_memory_test() {
    /* Construct the network */
    Network *network = new Network();
    Structure *main_structure = new Structure("working memory", PARALLEL);
    network->add_structure(main_structure);

    bool wrap = true;
    int num_cortical_regions = 3;

    int thal_ratio = 1;
    int cortex_size = 128;
    int thal_size = cortex_size / thal_ratio;

    float cortex_noise = 1.0;
    float cortex_noise_stdev = 0.3;

    bool exc_plastic = true;
    int exc_delay = 0;
    int inh_delay = 3;

    int ff_center = 15;
    int ff_surround = 19;
    int intra_cortex_center = 15;
    int intra_cortex_surround = 19;
    int gamma_center = 9;
    int gamma_surround = 15;

    float fraction = 0.1;

    // Feedforward circuit
    main_structure->add_layer(
        (new LayerConfig("feedforward",
            IZHIKEVICH, cortex_size, cortex_size,
            (new PropertyConfig())
            ->set_value("type", "poisson")))
        ->set(IZ_INIT, "regular"));

    // Thalamic relay
    main_structure->add_layer(
        (new LayerConfig("tl1_thalamus",
            IZHIKEVICH, 1, 1))
        ->set(IZ_INIT, "thalamo_cortical"));

    std::vector<Structure*> sub_structures;
    for (int i = 0 ; i < num_cortical_regions ; ++i) {
        Structure *sub_structure = new Structure(std::to_string(i), PARALLEL);

        // Thalamic gamma nucleus
        /*
        sub_structure->add_layer(
            (new LayerConfig("gamma_thalamus",
                IZHIKEVICH, thal_size, thal_size))
            ->set(IZ_INIT, "thalamo_cortical"));
        */

        // Cortical layers
        sub_structure->add_layer(
            (new LayerConfig("3_cortex",
                IZHIKEVICH, cortex_size, cortex_size,
                (new PropertyConfig())
                ->set_value("type", "normal")
                ->set_value("mean", std::to_string(cortex_noise))
                ->set_value("std_dev", std::to_string(cortex_noise_stdev))))
            ->set(IZ_INIT, "random positive"));
        /*
        sub_structure->add_layer(
            (new LayerConfig("6_cortex",
                IZHIKEVICH, cortex_size, cortex_size,
                (new NoiseConfig("normal"))
                ->set("mean", std::to_string(cortex_noise))
                ->set("std_dev", std::to_string(cortex_noise_stdev))))
            ->set(IZ_INIT, "regular"));
        */

        // Cortico-cortical connectivity
        /*
        sub_structure->connect_layers("3_cortex", "6_cortex",
            (new ConnectionConfig(exc_plastic, exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1, fraction)))
            ->set_arborized_config(new ArborizedConfig(intra_cortex_center,1,wrap)));
        */
        sub_structure->connect_layers("3_cortex", "3_cortex",
            (new ConnectionConfig(exc_plastic, exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1, fraction)))
            ->set_arborized_config(new ArborizedConfig(intra_cortex_center,1,wrap)));
        /*
        sub_structure->connect_layers("6_cortex", "6_cortex",
            (new ConnectionConfig(exc_plastic, exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1, fraction)))
            ->set_arborized_config(new ArborizedConfig(intra_cortex_center,1,wrap)));
        sub_structure->connect_layers("6_cortex", "3_cortex",
            (new ConnectionConfig(exc_plastic, exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1, fraction)))
            ->set_arborized_config(new ArborizedConfig(intra_cortex_center,1,wrap)));
        */

        if (intra_cortex_surround > intra_cortex_center) {
            /*
            sub_structure->connect_layers("3_cortex", "6_cortex",
                (new ConnectionConfig(false, inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(intra_cortex_center,
                        new FlatWeightConfig(0.1, fraction))))
                ->set_arborized_config(new ArborizedConfig(intra_cortex_surround,1,wrap)));
            */
            sub_structure->connect_layers("3_cortex", "3_cortex",
                (new ConnectionConfig(false, inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(intra_cortex_center,
                        new FlatWeightConfig(0.1, fraction))))
                ->set_arborized_config(new ArborizedConfig(intra_cortex_surround,1,wrap)));
            /*
            sub_structure->connect_layers("6_cortex", "6_cortex",
                (new ConnectionConfig(false, inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(intra_cortex_center,
                        new FlatWeightConfig(0.1, fraction))))
                ->set_arborized_config(new ArborizedConfig(intra_cortex_surround,1,wrap)));
            sub_structure->connect_layers("6_cortex", "3_cortex",
                (new ConnectionConfig(false, inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(intra_cortex_center,
                        new FlatWeightConfig(0.1, fraction))))
                ->set_arborized_config(new ArborizedConfig(intra_cortex_surround,1,wrap)));
            */
        }

        // Gamma connectivity
        /*
        sub_structure->connect_layers("gamma_thalamus", "6_cortex",
            (new ConnectionConfig(exc_plastic, 10 + exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1*thal_ratio, fraction)))
            ->set_arborized_config(new ArborizedConfig(gamma_center,1,wrap)));
        sub_structure->connect_layers("6_cortex", "gamma_thalamus",
            (new ConnectionConfig(exc_plastic, 10 + exc_delay, 0.5, CONVERGENT, ADD,
                new FlatWeightConfig(0.1*thal_ratio, fraction)))
            ->set_arborized_config(new ArborizedConfig(gamma_center,1,wrap)));

        if (gamma_surround > gamma_center) {
            sub_structure->connect_layers("gamma_thalamus", "6_cortex",
                (new ConnectionConfig(false, 10 + inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(gamma_center,
                        new FlatWeightConfig(0.1*thal_ratio, fraction))))
                ->set_arborized_config(new ArborizedConfig(gamma_surround,1,wrap)));
            sub_structure->connect_layers("6_cortex", "gamma_thalamus",
                (new ConnectionConfig(false, 10 + inh_delay, 0.5, CONVERGENT, SUB,
                    new SurroundWeightConfig(gamma_center,
                        new FlatWeightConfig(0.1*thal_ratio, fraction))))
                ->set_arborized_config(new ArborizedConfig(gamma_surround,1,wrap)));
        }
        */

        // Feedforward pathway
        if (i > 0) {
            Structure::connect(sub_structures[i-1], "3_cortex",
                sub_structure, "3_cortex",
                (new ConnectionConfig(exc_plastic, 0, 0.5, CONVERGENT, ADD,
                    new FlatWeightConfig(0.1*thal_ratio, fraction)))
                ->set_arborized_config(new ArborizedConfig(ff_center,1,wrap)));
            Structure::connect(sub_structure, "3_cortex",
                sub_structures[i-1], "3_cortex",
                (new ConnectionConfig(exc_plastic, 0, 0.5, CONVERGENT, ADD,
                    new FlatWeightConfig(0.1*thal_ratio, fraction)))
                ->set_arborized_config(new ArborizedConfig(ff_center,1,wrap)));
            if (ff_center > ff_surround) {
                Structure::connect(sub_structures[i-1], "3_cortex",
                    sub_structure, "3_cortex",
                    (new ConnectionConfig(false, 0, 0.5, CONVERGENT, SUB,
                        new SurroundWeightConfig(ff_center,
                            new FlatWeightConfig(0.1*thal_ratio, fraction))))
                    ->set_arborized_config(new ArborizedConfig(ff_surround,1,wrap)));
                Structure::connect(sub_structure, "3_cortex",
                    sub_structures[i-1], "3_cortex",
                    (new ConnectionConfig(false, 0, 0.5, CONVERGENT, SUB,
                        new SurroundWeightConfig(ff_center,
                            new FlatWeightConfig(0.1*thal_ratio, fraction))))
                    ->set_arborized_config(new ArborizedConfig(ff_surround,1,wrap)));
            }
        }

        // Thalamocortical control connectivity
        Structure::connect(main_structure, "tl1_thalamus",
            sub_structure, "3_cortex",
            (new ConnectionConfig(false, 0, 0.5, FULLY_CONNECTED, MULT,
                new FlatWeightConfig(0.1*thal_ratio)))
            ->set("myelinated", "true"));
        /*
        Structure::connect(main_structure, "tl1_thalamus",
            sub_structure, "6_cortex",
            (new ConnectionConfig(false, 0, 0.5, FULLY_CONNECTED, MULT,
                new FlatWeightConfig(0.1*thal_ratio)))
            ->set("myelinated", "true"));
        */


        network->add_structure(sub_structure);
        sub_structures.push_back(sub_structure);
    }

    // Sensory relay to cortex
    Structure::connect(main_structure, "feedforward",
        sub_structures[0], "3_cortex",
        (new ConnectionConfig(exc_plastic, 0, 0.5, CONVERGENT, ADD,
            new FlatWeightConfig(0.1*thal_ratio, fraction)))
        ->set_arborized_config(new ArborizedConfig(ff_center,1,wrap)));
    Structure::connect(main_structure, "feedforward",
        sub_structures[0], "3_cortex",
        (new ConnectionConfig(false, 0, 0.5, CONVERGENT, SUB,
            new SurroundWeightConfig(ff_center,
                new FlatWeightConfig(0.1*thal_ratio, fraction))))
        ->set_arborized_config(new ArborizedConfig(ff_surround,1,wrap)));

    // Modules
    auto env = new Environment();
    auto vis_mod = new ModuleConfig("visualizer");
    for (int i = 0 ; i < num_cortical_regions ; ++i)
        vis_mod
            //->add_layer(std::to_string(i), "6_cortex")
            //->add_layer(std::to_string(i), "gamma_thalamus")
            ->add_layer(std::to_string(i), "3_cortex");

    vis_mod->add_layer("working memory", "feedforward");
    env->add_module(vis_mod);

    env->add_module(
        (new ModuleConfig("periodic_input", "working memory", "tl1_thalamus"))
        ->set("random", "true")
        ->set("max", "3")
        ->set("rate", "500")
        ->set("verbose", "true"));

    auto c = new Context(network, env);
    Engine engine(c);
    print_network(network, env);
    delete engine.run(500000, true);
}

void dsst_test() {
    // Use default DSST parameters
    auto temp_config = new ModuleConfig("dsst");

    int rows = DSSTModule::get_input_rows(temp_config);
    int cols = DSSTModule::get_input_columns(temp_config);

    int cell_rows = DSSTModule::get_cell_rows(temp_config);
    int cell_cols = DSSTModule::get_cell_columns(temp_config);

    int focus_rows = rows - cell_rows;
    int focus_cols = cols - cell_cols;

    delete temp_config;

    /* Construct the network */
    Network *network = new Network();
    Structure *structure = new Structure("dsst", PARALLEL);
    network->add_structure(structure);

    structure->add_layer(
        (new LayerConfig("vision",
            "relay", rows, cols))
        ->set(IZ_INIT, "regular"));

    structure->add_layer(
        (new LayerConfig("what",
            "relay", cell_rows, cell_cols))
        ->set(IZ_INIT, "regular"));

    structure->add_layer(
        (new LayerConfig("focus",
            "relay", focus_rows, focus_cols))
        ->set(IZ_INIT, "regular"));

    structure->add_layer(
        (new LayerConfig("output_layer",
            "relay", 1, 1))
        ->set(IZ_INIT, "regular"));

    // Connect vision to what
    structure->set_second_order("what", "root");
    structure->connect_layers("vision", "what",
        (new ConnectionConfig(false, 0, 1, CONVOLUTIONAL, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            focus_rows,focus_cols,1,1,0,0)));
    structure->connect_layers("focus", "what",
        (new ConnectionConfig(false, 0, 1, CONVOLUTIONAL, MULT,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            focus_rows,focus_cols,0,0,0,0)));

    auto env = new Environment();
    env->add_module(
        (new ModuleConfig("dsst"))
        ->add_layer("dsst", "vision", "input")
        ->add_layer("dsst", "output_layer", "output"));

    env->add_module(
        (new ModuleConfig("visualizer"))
        ->add_layer("dsst", "vision")
        ->add_layer("dsst", "what")
        ->add_layer("dsst", "focus"));
    env->add_module(
        (new ModuleConfig("one_hot_random_input", "dsst", "focus"))
            ->set("rate", "10"));

    std::cout << "DSST test......\n";
    print_network(network, env);
    auto c = new Context(network, env);
    Engine engine(c);

    delete engine.run(1000000, true);
}

void debug_test() {
    /* Construct the network */
    Network *network = new Network();
    Structure *structure = new Structure("debug", PARALLEL);
    network->add_structure(structure);

    int rows = 10;
    int cols = 20;

    structure->add_layer(
        new LayerConfig("source",
            "debug", rows, cols));

    structure->add_layer(
        new LayerConfig("dest",
            "debug", rows, cols));

    // Check first order plastic connections
    structure->connect_layers("source", "dest",
        new ConnectionConfig(true, 0, 1, FULLY_CONNECTED, ADD));

    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, SUBSET, ADD))
        ->set_subset_config(new SubsetConfig(
            rows / 4, 3 * rows / 4,
            cols / 4, 3 * cols / 4,
            rows / 4, 3 * rows / 4,
            cols / 4, 3 * cols / 4)));

    structure->connect_layers("source", "dest",
        new ConnectionConfig(true, 0, 1, ONE_TO_ONE, ADD));

    // Non-wrapping standard arborized
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1)));
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, CONVOLUTIONAL, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1)));
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, DIVERGENT, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1)));

    // Non-wrapping unshifted arborized
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1,0,0)));
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, CONVOLUTIONAL, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1,0,0)));
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, DIVERGENT, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1,0,0)));

    // Wrapping standard arborized
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1,true)));
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, CONVOLUTIONAL, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1,true)));
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, DIVERGENT, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1,true)));

    // Wrapping unshifted arborized
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1,0,0,true)));
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, CONVOLUTIONAL, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1,0,0,true)));
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, DIVERGENT, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(new ArborizedConfig(
            3,3,1,1,0,0,true)));

    // Zero stride full size arborized
    // No divergent -- cannot have 0 stride
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, CONVERGENT, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(
            new ArborizedConfig(rows,cols,0,0,0,0)));
    structure->connect_layers("source", "dest",
        (new ConnectionConfig(true, 0, 1, CONVOLUTIONAL, ADD,
            new FlatWeightConfig(1.0)))
        ->set_arborized_config(
            new ArborizedConfig(rows,cols,0,0,0,0)));

    std::cout << "Debug test......\n";
    print_network(network);

    Engine engine(new Context(network));
    delete engine.run(1, true);
}

int cli() {
    bool quit = false;
    Engine *engine = nullptr;
    Context *context = nullptr;

    while (not quit) {
        std::cout << "Options:" << std::endl;
        std::cout << "Load (N)etwork" << std::endl;
        if (context != nullptr) {
            std::cout << "Load (E)nvironment" << std::endl;
            std::cout << "Load (S)tate" << std::endl;
            std::cout << "(R)un Engine" << std::endl;
        }
        std::cout << "(Q)uit" << std::endl;

        std::cout << std::endl << "Enter option: ";

        std::string input;
        std::cin >> input;
        try {
            switch (input.at(0)) {
                case 'n':
                case 'N':
                    std::cout << "Enter network name: ";
                    std::cin >> input;
                    context = new Context(load_network(input + ".json"));
                    break;
                case 'q':
                case 'Q':
                    quit = true;
                    break;
                default:
                    if (context == nullptr) throw std::runtime_error("");
                    switch (input.at(0)) {
                        case 'e':
                        case 'E':
                            std::cout << "Enter environment name: ";
                            std::cin >> input;
                            context->set_environment(
                                load_environment(input + ".json"));
                            if (engine != nullptr)
                                engine->rebuild();
                            break;
                        case 's':
                        case 'S':
                            std::cout << "Enter state name: ";
                            std::cin >> input;
                            context->get_state()->load(input + ".bin");
                            if (engine != nullptr)
                                engine->rebuild();
                            break;
                        case 'r':
                        case 'R':
                            std::cout << "Number of iterations: ";
                            std::cin >> input;
                            int iterations = stoi(input);
                            try {
                                if (engine == nullptr)
                                    engine = new Engine(context);
                                engine->run(iterations, true);
                            } catch (std::runtime_error e) {
                                printf("Fatal error -- exiting...\n");
                                return 1;
                            }
                            break;
                    }
            }
        } catch (...) {
            std::cout << "Unrecognized input!" << std::endl;
        }
        std::cout << std::endl;
    }
    return 0;
}


int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(nullptr));

    // Suppress warnings
    ErrorManager::get_instance()->set_warnings(false);

    return cli();

    /*
    try {
        //mnist_test();
        mnist_perceptron_test();
        //old_test();
        //simple_test();
        //single_field_test();
        //game_of_life_test();
        //working_memory_test();
        //dsst_test();
        //debug_test();

        return 0;
    } catch (std::runtime_error e) {
        printf("Fatal error -- exiting...\n");
        return 1;
    }
    */
}
