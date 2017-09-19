#include <cstdlib>
#include <iostream>

#include "io/impl/one_step_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#include <sstream>
#include <iostream>

REGISTER_MODULE(OneStepInputModule, "one_step_input", INPUT);

OneStepInputModule::OneStepInputModule(LayerList layers, ModuleConfig *config)
        : Module(layers), active(true), cleared(false) {
    enforce_equal_layer_sizes("one_step_input");

    float max_value = std::stof(config->get_property("max", "1.0"));
    float fraction = std::stof(config->get_property("fraction", "1.0"));
    bool uniform = config->get_property("uniform", "false") == "true";
    this->verbose = config->get_property("verbose", "false") == "true";

    if (max_value <= 0.0)
        ErrorManager::get_instance()->log_error(
            "Invalid max value for one step input generator!");
    if (fraction <= 0.0 or fraction > 1.0)
        ErrorManager::get_instance()->log_error(
            "Invalid fraction for one step input generator!");

    this->random_values = (float*) malloc (layers.at(0)->size * sizeof(float));

    if (uniform)
        fSet(this->random_values, layers.at(0)->size, max_value, fraction);
    else
        fRand(this->random_values, layers.at(0)->size, 0.0, max_value, fraction);
}

OneStepInputModule::~OneStepInputModule() {
    free(this->random_values);
}

void OneStepInputModule::feed_input(Buffer *buffer) {
    for (auto layer : layers) {
        if (active) {
            float *input = buffer->get_input(layer);
            for (int nid = 0 ; nid < layer->size; ++nid)
                input[nid] = this->random_values[nid];
            if (verbose) {
                for (int nid = 0 ; nid < layer->size; ++nid)
                    std::cout << this->random_values[nid];
                std::cout << std::endl;
            }
            buffer->set_dirty(layer);
            active = false;
        } else if (not cleared) {
            float *input = buffer->get_input(layer);
            fSet(input, layer->size, 0.0);
            buffer->set_dirty(layer);
            cleared = true;
        }
    }
}
