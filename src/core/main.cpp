#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "builder.h"
#include "context.h"
#include "network/network.h"
#include "state/state.h"
#include "io/module.h"
#include "io/impl/dsst_module.h"
#include "io/environment.h"
#include "engine/engine.h"
#include "util/property_config.h"

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
    engine.run(PropertyConfig({{"iterations", "60000"},
                               {"worker threads", "16"},
                               {"multithreaded", "false"},
                               {"verbose", "true"},
                               {"devices", "2"}}));

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
    engine.run(PropertyConfig(
        {{"iterations", "10000"},
         {"devices", "0"},
         {"worker threads", "16"},
         {"multithreaded", "true"},
         {"verbose", "true"},
         {"learning flag", "false"}}));
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
    // Suppress warnings
    Logger::warnings = false;

    // Set single GPU
    //ResourceManager::get_instance()->set_gpu(0);

    //mnist_test();
    //working_memory_test();
    return cli();

    /*
    try {
        mnist_test();
        return 0;
    } catch (std::runtime_error e) {
        printf("Fatal error -- exiting...\n");
        return 1;
    }
    */
}
