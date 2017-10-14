#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "builder.h"
#include "context.h"
#include "network/network.h"
#include "state/state.h"
#include "io/module.h"
#include "io/impl/dsst_module.h"
#include "io/environment.h"
#include "engine/engine.h"
#include "util/tools.h"
#include "util/property_config.h"

#define IZHIKEVICH "izhikevich"
#define HEBBIAN_RATE_ENCODING "hebbian_rate_encoding"
#define BACKPROP_RATE_ENCODING "backprop_rate_encoding"
#define RELAY "relay"

#define IZ_INIT "init"

void print_network(Network *network) {
    printf("Built network.\n");
    printf("  - neurons     : %10d\n", network->get_num_neurons());
    printf("  - layers      : %10d\n", network->get_num_layers());
    printf("  - connections : %10d\n", network->get_num_connections());
    printf("  - weights     : %10d\n", network->get_num_weights());

    for (auto structure : network->get_structures()) {
        for (auto layer : structure->get_layers()) {
            printf("%-20s   \n",
                (layer->structure->name + "->" + layer->name).c_str());
        }
        std::cout << std::endl;
    }
}

void mnist_test() {
    NetworkConfig *network_config = new NetworkConfig();
    StructureConfig *structure = new StructureConfig("mnist", FEEDFORWARD);

    structure->add_layer(
        (new PropertyConfig())
            ->set("name", "input_layer")
            ->set("neural model", "relay")
            ->set("rows", "28")
            ->set("columns", "28"));
    structure->add_layer(
        (new PropertyConfig())
            ->set("name", "output_layer")
            ->set("neural model", "perceptron")
            ->set("rows", "1")
            ->set("columns", "10"));
    structure->add_layer(
        (new PropertyConfig())
            ->set("name", "bias_layer")
            ->set("neural model", "relay")
            ->set("rows", "1")
            ->set("columns", "1"));

    network_config->add_structure(structure);

    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "fully connected")
            ->set("opcode", "add")
            ->set_child("weight config",
                (new PropertyConfig())
                    ->set("type", "flat")
                    ->set("weight", "0"))
            ->set("from structure", "mnist")
            ->set("to structure", "mnist")
            ->set("from layer", "input_layer")
            ->set("to layer", "output_layer"));

    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "fully connected")
            ->set("opcode", "add")
            ->set_child("weight config",
                (new PropertyConfig())
                    ->set("type", "flat")
                    ->set("weight", "0"))
            ->set("from structure", "mnist")
            ->set("to structure", "mnist")
            ->set("from layer", "bias_layer")
            ->set("to layer", "output_layer"));

    auto network = new Network(network_config);

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
    auto state = new State(network);
    Engine engine(Context(network, env, state));
    engine.run(PropertyConfig({{"iterations", "60000"}}));

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
    engine.run(PropertyConfig(
        {{"iterations", "10000"},
         {"learning flag", "false"}}));
}

void game_of_life_test() {
    NetworkConfig *network_config = new NetworkConfig();
    StructureConfig *structure = new StructureConfig("game_of_life", PARALLEL);

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
    float random_fraction = 0.5;
    float rate = 1000;

    structure->add_layer(
        (new PropertyConfig())
            ->set("name", "board")
            ->set("neural model", "game_of_life")
            ->set("rows", board_dim)
            ->set("columns", board_dim)
            ->set_child("noise config",
                (new PropertyConfig())
                    ->set("type", "poisson")
                    ->set("value", birth_min)
                    ->set("rate", "0.5"))
            ->set("survival_min", survival_min)
            ->set("survival_max", survival_max)
            ->set("birth_min", birth_min)
            ->set("birth_max", birth_max));

    network_config->add_structure(structure);

    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "false")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("convolutional", true)
            ->set("opcode", "add")
            ->set_child("weight config",
                (new PropertyConfig())
                    ->set("type", "surround")
                    ->set("size", "1")
                    ->set("child type", "flat")
                    ->set("weight", "1"))
            ->set("from structure", "game_of_life")
            ->set("to structure", "game_of_life")
            ->set("from layer", "board")
            ->set("to layer", "board")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", neighborhood_size)
                    ->set("wrap", (wrap) ? "true" : "false")));

    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "false")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("convolutional", true)
            ->set("opcode", "sub")
            ->set_child("weight config",
                (new PropertyConfig())
                    ->set("type", "surround")
                    ->set("rows", neighborhood_size)
                    ->set("columns", neighborhood_size)
                    ->set("child type", "flat")
                    ->set("weight", "1"))
            ->set("from structure", "game_of_life")
            ->set("to structure", "game_of_life")
            ->set("from layer", "board")
            ->set("to layer", "board")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", neighborhood_size+2)
                    ->set("offset", (-neighborhood_size+1)/2)
                    ->set("wrap", (wrap) ? "true" : "false")));

    auto network = new Network(network_config);

    // Modules
    auto env = new Environment();

    // Refresh state
    env->add_module(
        (new ModuleConfig("periodic_input", "game_of_life", "board"))
            ->set("max", birth_min)
            ->set("rate", rate)
            ->set("clear", "true")
            ->set("verbose", "false")
            ->set("fraction", random_fraction));

    env->add_module(
        new ModuleConfig("visualizer", "game_of_life", "board"));

    auto state = new State(network);
    Engine engine(Context(network, env, state));
    print_network(network);
    engine.run(PropertyConfig({{"iterations", "500000"}}));
    delete network;
    delete env;
    delete state;
}

void working_memory_test() {
    NetworkConfig *network_config = new NetworkConfig();
    StructureConfig *main_structure = new StructureConfig("working memory", PARALLEL);

    bool wrap = true;
    int num_cortical_regions = 1;

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

    float fraction = 0.1;

    // Feedforward circuit
    main_structure->add_layer(
        (new PropertyConfig())
            ->set("name", "feedforward")
            ->set("neural model", IZHIKEVICH)
            ->set("rows", cortex_size)
            ->set("columns", cortex_size)
            ->set_child("noise config",
                (new PropertyConfig())
                    ->set("type", "poisson")
                    ->set("rate", 1))
            ->set(IZ_INIT, "regular"));

    // Thalamic relay
    /*
    main_structure->add_layer(
        (new PropertyConfig())
            ->set("name", "tl1_thalamus")
            ->set("neural model", IZHIKEVICH)
            ->set("rows", "1")
            ->set("columns", "1")
            ->set(IZ_INIT, "thalamo_cortical"));
    */

    for (int i = 0 ; i < num_cortical_regions ; ++i) {
        StructureConfig *sub_structure = new StructureConfig(std::to_string(i), PARALLEL);

        // Cortical layers
        sub_structure->add_layer(
            (new PropertyConfig())
                ->set("name", "3_cortex")
                ->set("neural model", IZHIKEVICH)
                ->set("rows", cortex_size)
                ->set("columns", cortex_size)
                ->set(IZ_INIT, "random positive"));

        // Cortico-cortical connectivity
        network_config->add_connection(
            (new PropertyConfig())
                ->set("plastic", (exc_plastic) ? "true" : "false")
                ->set("delay", exc_delay)
                ->set("max weight", "0.5")
                ->set("type", "convergent")
                ->set("opcode", "add")
                ->set_child("weight config",
                    (new PropertyConfig())
                        ->set("type", "power law")
                        ->set("exponent", "1.5")
                        ->set("fraction", fraction))
                ->set("from structure", i)
                ->set("to structure", i)
                ->set("from layer", "3_cortex")
                ->set("to layer", "3_cortex")
                ->set_child("arborized config",
                    (new PropertyConfig())
                        ->set("field size", intra_cortex_center)
                        ->set("wrap", (wrap) ? "true" : "false")));

        if (intra_cortex_surround > intra_cortex_center) {
            network_config->add_connection(
                (new PropertyConfig())
                    ->set("plastic", "false")
                    ->set("delay", inh_delay)
                    ->set("max weight", "0.5")
                    ->set("type", "convergent")
                    ->set("opcode", "sub")
                    ->set_child("weight config",
                        (new PropertyConfig())
                            ->set("type", "surround")
                            ->set("size", intra_cortex_center)
                            ->set("child type", "flat")
                            ->set("weight", "0.1")
                            ->set("fraction", fraction))
                    ->set("from structure", i)
                    ->set("to structure", i)
                    ->set("from layer", "3_cortex")
                    ->set("to layer", "3_cortex")
                    ->set_child("arborized config",
                        (new PropertyConfig())
                            ->set("field size", intra_cortex_center)
                            ->set("wrap", (wrap) ? "true" : "false")));
        }

        // Feedforward pathway
        if (i > 0) {
            network_config->add_connection(
                (new PropertyConfig())
                    ->set("plastic", (exc_plastic) ? "true" : "false")
                    ->set("delay", "0")
                    ->set("max weight", 0.5 * thal_ratio)
                    ->set("type", "convergent")
                    ->set("opcode", "add")
                    ->set_child("weight config",
                        (new PropertyConfig())
                            ->set("type", "power law")
                            ->set("exponent", "1.5")
                            ->set("fraction", fraction))
                    ->set("from structure", i-1)
                    ->set("to structure", i)
                    ->set("from layer", "3_cortex")
                    ->set("to layer", "3_cortex")
                    ->set_child("arborized config",
                        (new PropertyConfig())
                            ->set("field size", ff_center)
                            ->set("wrap", (wrap) ? "true" : "false")));
            network_config->add_connection(
                (new PropertyConfig())
                    ->set("plastic", (exc_plastic) ? "true" : "false")
                    ->set("delay", "0")
                    ->set("max weight", 0.5 * thal_ratio)
                    ->set("type", "convergent")
                    ->set("opcode", "add")
                    ->set_child("weight config",
                        (new PropertyConfig())
                            ->set("type", "power law")
                            ->set("exponent", "1.5")
                            ->set("fraction", fraction))
                    ->set("from structure", i)
                    ->set("to structure", i-1)
                    ->set("from layer", "3_cortex")
                    ->set("to layer", "3_cortex")
                    ->set_child("arborized config",
                        (new PropertyConfig())
                            ->set("field size", ff_center)
                            ->set("wrap", (wrap) ? "true" : "false")));
            if (ff_center > ff_surround) {
                network_config->add_connection(
                    (new PropertyConfig())
                        ->set("plastic", "false")
                        ->set("delay", "0")
                        ->set("max weight", "0.5")
                        ->set("type", "convergent")
                        ->set("opcode", "sub")
                        ->set_child("weight config",
                            (new PropertyConfig())
                                ->set("type", "surround")
                                ->set("size", ff_center)
                                ->set("child type", "flat")
                                ->set("weight", 0.1*thal_ratio)
                                ->set("fraction", fraction))
                        ->set("from structure", i)
                        ->set("to structure", i-1)
                        ->set("from layer", "3_cortex")
                        ->set("to layer", "3_cortex")
                        ->set_child("arborized config",
                            (new PropertyConfig())
                                ->set("field size", ff_surround)
                                ->set("wrap", (wrap) ? "true" : "false")));
                network_config->add_connection(
                    (new PropertyConfig())
                        ->set("plastic", "false")
                        ->set("delay", "0")
                        ->set("max weight", "0.5")
                        ->set("type", "convergent")
                        ->set("opcode", "sub")
                        ->set_child("weight config",
                            (new PropertyConfig())
                                ->set("type", "surround")
                                ->set("size", ff_center)
                                ->set("child type", "flat")
                                ->set("weight", 0.1*thal_ratio)
                                ->set("fraction", fraction))
                        ->set("from structure", i-1)
                        ->set("to structure", i)
                        ->set("from layer", "3_cortex")
                        ->set("to layer", "3_cortex")
                        ->set_child("arborized config",
                            (new PropertyConfig())
                                ->set("field size", ff_surround)
                                ->set("wrap", (wrap) ? "true" : "false")));
            }
        }

        // Thalamocortical control connectivity
        /*
        network_config->add_connection(
            (new PropertyConfig())
                ->set("plastic", "false")
                ->set("delay", "0")
                ->set("max weight", "0.5")
                ->set("type", "fully connected")
                ->set("opcode", "mult")
                ->set_child("weight config",
                    (new PropertyConfig())
                        ->set("type", "flat")
                        ->set("weight", 0.1 * thal_ratio))
                ->set("from structure", "working memory")
                ->set("to structure", i)
                ->set("from layer", "tl1_thalamus")
                ->set("to layer", "3_cortex")
                ->set("myelinated", "true"));
        */


        network_config->add_structure(sub_structure);
    }

    // Sensory relay to cortex
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", (exc_plastic) ? "true" : "false")
            ->set("delay", "0")
            ->set("max weight", "0.5")
            ->set("type", "convergent")
            ->set("opcode", "add")
            ->set_child("weight config",
                (new PropertyConfig())
                    ->set("type", "flat")
                    ->set("weight", 0.1 * thal_ratio)
                    ->set("fraction", fraction))
            ->set("from structure", "working memory")
            ->set("to structure", "0")
            ->set("from layer", "feedforward")
            ->set("to layer", "3_cortex")
            ->set("myelinated", "true")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", ff_center)
                    ->set("wrap", (wrap) ? "true" : "false")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "false")
            ->set("delay", "0")
            ->set("max weight", "0.5")
            ->set("type", "convergent")
            ->set("opcode", "sub")
            ->set_child("weight config",
                (new PropertyConfig())
                    ->set("type", "surround")
                    ->set("size", ff_center)
                    ->set("child type", "flat")
                    ->set("weight", 0.1*thal_ratio)
                    ->set("fraction", fraction))
            ->set("from structure", "working memory")
            ->set("to structure", "0")
            ->set("from layer", "feedforward")
            ->set("to layer", "3_cortex")
            ->set("myelinated", "true")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", ff_surround)
                    ->set("wrap", (wrap) ? "true" : "false")));

    network_config->add_structure(main_structure);
    auto network = new Network(network_config);

    // Modules
    auto env = new Environment();
    auto vis_mod = new ModuleConfig("visualizer");
    for (int i = 0 ; i < num_cortical_regions ; ++i)
        vis_mod
            //->add_layer(std::to_string(i), "6_cortex")
            ->add_layer(std::to_string(i), "3_cortex");

    vis_mod->add_layer("working memory", "feedforward");
    env->add_module(vis_mod);

    /*
    env->add_module(
        (new ModuleConfig("periodic_input", "working memory", "tl1_thalamus"))
        ->set("random", "true")
        ->set("max", "3")
        ->set("rate", "500")
        ->set("verbose", "true"));
    */

    auto state = new State(network);
    Engine engine(Context(network, env, state));
    print_network(network);
    auto context = engine.run(PropertyConfig({{"verbose", "true"}}));
    delete network;
    delete env;
    delete state;
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
    NetworkConfig *network_config = new NetworkConfig();
    StructureConfig *structure = new StructureConfig("dsst", PARALLEL);

    structure->add_layer(
        (new PropertyConfig())
            ->set("name", "vision")
            ->set("neural model", "relay")
            ->set("rows", rows)
            ->set("columns", cols));

    structure->add_layer(
        (new PropertyConfig())
            ->set("name", "what")
            ->set("neural model", "relay")
            ->set("rows", cell_rows)
            ->set("columns", cell_cols)
            ->set_array("dendrites",
                {
                    (new PropertyConfig())
                        ->set("name", "fixation")
                        ->set("second order", "true")
                }
            ));

    structure->add_layer(
        (new PropertyConfig())
            ->set("name", "focus")
            ->set("neural model", "relay")
            ->set("rows", focus_rows)
            ->set("columns", focus_cols));

    structure->add_layer(
        (new PropertyConfig())
            ->set("name", "output_layer")
            ->set("neural model", "relay")
            ->set("rows", "1")
            ->set("columns", "1"));

    // Connect vision to what
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "false")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("convolutional", true)
            ->set("opcode", "add")
            ->set_child("weight config",
                (new PropertyConfig())
                    ->set("type", "flat")
                    ->set("weight", "1"))
            ->set("from structure", "dsst")
            ->set("to structure", "dsst")
            ->set("from layer", "vision")
            ->set("to layer", "what")
            ->set("dendrite", "fixation")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("row field size", focus_rows)
                    ->set("column field size", focus_cols)
                    ->set("offset", "0")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "false")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("convolutional", true)
            ->set("opcode", "mult")
            ->set_child("weight config",
                (new PropertyConfig())
                    ->set("type", "flat")
                    ->set("weight", "1"))
            ->set("from structure", "dsst")
            ->set("to structure", "dsst")
            ->set("from layer", "focus")
            ->set("to layer", "what")
            ->set("dendrite", "fixation")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("row field size", focus_rows)
                    ->set("column field size", focus_cols)
                    ->set("stride", "0")
                    ->set("offset", "0")));

    network_config->add_structure(structure);
    auto network = new Network(network_config);

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
    auto state = new State(network);
    Engine engine(Context(network, env, state));
    print_network(network);
    auto context = engine.run(PropertyConfig({{"verbose", "true"}}));
    delete network;
    delete env;
    delete state;
}

void debug_test() {
    NetworkConfig *network_config = new NetworkConfig();
    StructureConfig *structure = new StructureConfig("debug", PARALLEL);

    int rows = 10;
    int cols = 20;

    structure->add_layer(
        (new PropertyConfig())
            ->set("name", "source")
            ->set("neural model", "debug")
            ->set("rows", rows)
            ->set("columns", cols));
    structure->add_layer(
        (new PropertyConfig())
            ->set("name", "dest")
            ->set("neural model", "debug")
            ->set("rows", rows)
            ->set("columns", cols));

    network_config->add_structure(structure);

    // Check first order plastic connections
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "fully connected")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest"));

    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "subset")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("subset config",
                (new PropertyConfig())
                    ->set("from row start", rows / 4)
                    ->set("from row end",   3 * rows / 4)
                    ->set("to row start",   rows / 4)
                    ->set("to row end",     3 * rows / 4)
                    ->set("from column start", cols / 4)
                    ->set("from column end",   3 * cols / 4)
                    ->set("to column start",   cols / 4)
                    ->set("to column end",     3 * cols / 4)));

    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "one to one")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest"));

    // Non-wrapping standard arborized
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("convolutional", true)
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "divergent")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")));

    // Non-wrapping unshifted arborized
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")
                    ->set("offset", "0")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("convolutional", true)
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")
                    ->set("offset", "0")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "divergent")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")
                    ->set("offset", "0")));

    // Wrapping standard arborized
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")
                    ->set("wrap", "true")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("convolutional", true)
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")
                    ->set("wrap", "true")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "divergent")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")
                    ->set("wrap", "true")));

    // Wrapping unshifted arborized
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")
                    ->set("offset", "0")
                    ->set("wrap", "true")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("convolutional", true)
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")
                    ->set("offset", "0")
                    ->set("wrap", "true")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "divergent")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("field size", "3")
                    ->set("stride", "1")
                    ->set("offset", "0")
                    ->set("wrap", "true")));

    // Zero stride full size arborized
    // No divergent -- cannot have 0 stride
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("row field size", rows)
                    ->set("column field size", cols)
                    ->set("stride", "0")
                    ->set("offset", "0")
                    ->set("wrap", "true")));
    network_config->add_connection(
        (new PropertyConfig())
            ->set("plastic", "true")
            ->set("delay", "0")
            ->set("max weight", "1")
            ->set("type", "convergent")
            ->set("convolutional", true)
            ->set("opcode", "add")
            ->set("from structure", "debug")
            ->set("to structure", "debug")
            ->set("from layer", "source")
            ->set("to layer", "dest")
            ->set_child("arborized config",
                (new PropertyConfig())
                    ->set("row field size", rows)
                    ->set("column field size", cols)
                    ->set("stride", "0")
                    ->set("offset", "0")
                    ->set("wrap", "true")));

    auto network = new Network(network_config);

    std::cout << "Debug test......\n";
    print_network(network);

    auto state = new State(network);
    Engine engine(Context(network, nullptr, state));
    auto context = engine.run(PropertyConfig({{"iterations", "1"}}));
    delete network;
    delete state;
}

int cli() {
    bool quit = false;
    Network *network = nullptr;
    Environment *environment = nullptr;
    State *state = nullptr;
    Engine *engine = nullptr;

    while (not quit) {
        std::cout << "Options:" << std::endl;
        std::cout << "Load (N)etwork" << std::endl;
        if (network != nullptr) {
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
                    network = load_network(input + ".json");
                    state = new State(network);
                    break;
                case 'q':
                case 'Q':
                    quit = true;
                    break;
                default:
                    if (network == nullptr) throw std::invalid_argument("");
                    switch (input.at(0)) {
                        case 'e':
                        case 'E':
                            std::cout << "Enter environment name: ";
                            std::cin >> input;
                            environment = load_environment(input + ".json");
                            if (engine != nullptr)
                                engine->rebuild();
                            break;
                        case 's':
                        case 'S':
                            std::cout << "Enter state name: ";
                            std::cin >> input;
                            state->load(input + ".bin");
                            if (engine != nullptr)
                                engine->rebuild();
                            break;
                        case 'r':
                        case 'R':
                            std::cout << "Number of iterations: ";
                            std::cin >> input;
                            try {
                                if (engine == nullptr)
                                    engine = new Engine(Context(network, environment, state));
                                engine->run(PropertyConfig({{"iterations", input}}));
                            } catch (std::runtime_error e) {
                                printf("Fatal error -- exiting...\n");
                                return 1;
                            }
                            break;
                    }
            }
        } catch (std::invalid_argument) {
            std::cout << "Unrecognized input!" << std::endl;
        }
        std::cout << std::endl;
    }
    if (network != nullptr) delete network;
    if (environment != nullptr) delete environment;
    if (state != nullptr) delete state;
    if (engine != nullptr) delete engine;
    return 0;
}


int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(nullptr));

    // Suppress warnings
    ErrorManager::warnings = false;

    working_memory_test();
    return cli();

    /*
    try {
        mnist_test();
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
