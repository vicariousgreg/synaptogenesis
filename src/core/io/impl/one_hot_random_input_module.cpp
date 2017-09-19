#include <cstdlib>

#include "io/impl/one_hot_random_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#include <sstream>
#include <iostream>

REGISTER_MODULE(OneHotRandomInputModule, "one_hot_random_input");

static void shuffle(float *vals, float max, int size, bool verbose) {
    int random_index = rand() % size;
    for (int nid = 0 ; nid < size; ++nid) {
        /*  Randomly selects one input to activate */
        vals[nid] =  (nid == random_index) ? max : 0;
        if (verbose) std::cout << vals[nid] << " ";
    }
    if (verbose) std::cout << std::endl;
}

OneHotRandomInputModule::OneHotRandomInputModule(LayerList layers, ModuleConfig *config)
        : Module(layers), timesteps(0) {
    enforce_equal_layer_sizes("one_hot_random_input");
    set_io_type(INPUT);

    this->max_value = std::stof(config->get_property("max", "1.0"));
    this->shuffle_rate = std::stoi(config->get_property("rate", "100"));
    this->end = std::stoi(config->get_property("end", "0"));
    this->verbose = config->get_property("verbose", "true") == "true";

    if (this->max_value <= 0.0)
        ErrorManager::get_instance()->log_error(
            "Invalid max value for random input generator!");
    if (this->shuffle_rate <= 0)
        ErrorManager::get_instance()->log_error(
            "Invalid shuffle rate for random input generator!");
    if (this->end < 0)
        ErrorManager::get_instance()->log_error(
            "Invalid shuffle rate for random input generator!");

    this->random_values = (float*) malloc (layers.at(0)->size * sizeof(float));
}

OneHotRandomInputModule::~OneHotRandomInputModule() {
    free(this->random_values);
}

void OneHotRandomInputModule::feed_input(Buffer *buffer) {
    for (auto layer : layers) {
        if (end == 0 or timesteps <= end and timesteps % shuffle_rate == 0) {
            if (verbose) {
                std::cout << "============================ SHUFFLE\n";
                if (end != 0) std::cout << "  *  ";
            }

            // Shuffle
            int random_index = rand() % layer->size;
            for (int nid = 0 ; nid < layer->size; ++nid) {
                /*  Randomly selects one input to activate */
                random_values[nid] =  (nid == random_index) ? max_value : 0;
                if (verbose) std::cout << random_values[nid] << " ";
            }
            if (verbose) std::cout << std::endl;

            float *input = buffer->get_input(layer);
            for (int nid = 0 ; nid < layer->size; ++nid)
                input[nid] = this->random_values[nid];
            buffer->set_dirty(layer);
        } else if (timesteps == end) {
            if (verbose)
                std::cout << "========================================== CLEAR\n";
            float *input = buffer->get_input(layer);
            for (int nid = 0 ; nid < layer->size; ++nid)
                input[nid] = 0.0;
            buffer->set_dirty(layer);
        }
    }
}

void OneHotRandomInputModule::cycle() {
    if (end == 0 or timesteps <= end) ++timesteps;
}
