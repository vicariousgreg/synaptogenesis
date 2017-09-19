#include <cstdlib>

#include "io/impl/one_hot_cyclic_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#include <sstream>
#include <iostream>

REGISTER_MODULE(OneHotCyclicInputModule, "one_hot_cyclic_input", INPUT);

OneHotCyclicInputModule::OneHotCyclicInputModule(LayerList layers, ModuleConfig *config)
        : Module(layers), timesteps(0), index(layers.at(0)->size-1) {
    this->max_value = std::stof(config->get_property("max", "1.0"));
    this->cycle_rate = std::stoi(config->get_property("rate", "100"));
    this->end = std::stoi(config->get_property("end", "0"));

    if (this->max_value <= 0.0)
        ErrorManager::get_instance()->log_error(
            "Invalid max value for random input generator!");
    if (this->cycle_rate <= 0)
        ErrorManager::get_instance()->log_error(
            "Invalid cycle rate for cyclic input generator!");
    if (this->end < 0)
        ErrorManager::get_instance()->log_error(
            "Invalid endpoint for cyclic input generator!");
}

void OneHotCyclicInputModule::feed_input(Buffer *buffer) {
    if (timesteps <= end) {
        // Cycle if necessary
        if (timesteps % cycle_rate == 0) {
            std::cout << "============================ SHUFFLE\n";
            if (end != 0) std::cout << "  *  ";

            // Cycle
            float *input = buffer->get_input(this->layers.at(0));
            input[index] = 0.0;
            index = (index + 1) % this->layers.at(0)->size;
            input[index] = max_value;
            buffer->set_dirty(this->layers.at(0));

            // Print
            for (int nid = 0 ; nid < this->layers.at(0)->size; ++nid)
                    std::cout << ((nid == index) ? max_value : 0) << " ";
            std::cout << std::endl;
        }

        // Clear or update
        if (timesteps == end) {
            std::cout << "========================================== CLEAR\n";

            // Clear
            float *input = buffer->get_input(this->layers.at(0));
            for (int nid = 0 ; nid < this->layers.at(0)->size; ++nid)
                input[nid] = 0.0;
            buffer->set_dirty(this->layers.at(0));
        }
    }
}

void OneHotCyclicInputModule::cycle() {
    if (timesteps <= end) ++timesteps;
}
