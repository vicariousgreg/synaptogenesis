#include <cstdlib>

#include "io/module/noise_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#include <iostream>

NoiseInputModule::NoiseInputModule(Layer *layer, std::string params)
        : Module(layer) {
    this->max_value = strtof(params.c_str(), NULL);
    if (this->max_value == 0.0)
        ErrorManager::get_instance()->log_error(
            "Invalid max value for random input generator!");
}

void NoiseInputModule::feed_input(Buffer *buffer) {
    int offset = this->layer->input_index;
    float *input = buffer->get_input();
    for (int nid = 0 ; nid < this->layer->size; ++nid) {
        input[offset + nid] = fRand(0, this->max_value);
    }
}
