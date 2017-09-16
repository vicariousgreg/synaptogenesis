#include <cstdlib>

#include "io/module/impl/random_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#include <sstream>
#include <iostream>

REGISTER_MODULE(RandomInputModule, "random_input", INPUT);

RandomInputModule::RandomInputModule(Layer *layer, ModuleConfig *config)
        : Module(layer), timesteps(0) {
    this->max_value = std::stof(config->get_property("max", "1.0"));
    this->shuffle_rate = std::stoi(config->get_property("rate", "1"));
    this->fraction = std::stof(config->get_property("fraction", "1.0"));
    this->uniform = config->get_property("uniform", "false") == "true";
    this->clear = config->get_property("clear", "false") == "true";
    this->verbose = config->get_property("verbose", "false") == "true";

    if (this->max_value <= 0.0)
        ErrorManager::get_instance()->log_error(
            "Invalid max value for random input generator!");
    if (this->shuffle_rate <= 0)
        ErrorManager::get_instance()->log_error(
            "Invalid shuffle rate for random input generator!");
    if (this->fraction <= 0.0 or this->fraction > 1.0)
        ErrorManager::get_instance()->log_error(
            "Invalid fraction for random input generator!");

    this->random_values = (float*) malloc (layer->size * sizeof(float));
}

RandomInputModule::~RandomInputModule() {
    free(this->random_values);
}

void RandomInputModule::feed_input(Buffer *buffer) {
    if (timesteps++ % shuffle_rate == 0) {
        if (uniform)
            fSet(random_values, layer->size, max_value, fraction);
        else
            fRand(random_values, layer->size, 0.0, max_value, fraction);
        float *input = buffer->get_input(this->layer);
        for (int nid = 0 ; nid < this->layer->size; ++nid)
            input[nid] = this->random_values[nid];

        if (verbose) {
            std::cout << "============================ SHUFFLE\n";
            for (int nid = 0 ; nid < this->layer->size; ++nid)
                std::cout << this->random_values[nid] << " ";
            std::cout << std::endl;
        }

        buffer->set_dirty(this->layer);
    } else if (clear and (timesteps % shuffle_rate == 2)) {
        if (verbose) std::cout << "============================ CLEAR\n";
        fSet(random_values, layer->size, 0.0);

        float *input = buffer->get_input(this->layer);
        for (int nid = 0 ; nid < this->layer->size; ++nid)
            input[nid] = this->random_values[nid];
        buffer->set_dirty(this->layer);
    }
}
