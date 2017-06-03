#include <cstdlib>

#include "io/module/one_hot_random_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#include <sstream>
#include <iostream>

REGISTER_MODULE(OneHotRandomInputModule, "one_hot_random_input", INPUT);

static void shuffle(float *vals, float max, int size) {
    int random_index = rand() % size;
    for (int nid = 0 ; nid < size; ++nid) {
        /*  Randomly selects one input to activate */
        vals[nid] =  (nid == random_index) ? max : 0;
        std::cout << vals[nid] << " ";
    }
    std::cout << std::endl;
}

OneHotRandomInputModule::OneHotRandomInputModule(Layer *layer, std::string params)
        : Module(layer), timesteps(0) {
    std::stringstream stream(params);
    if (!stream.eof()) {
        stream >> this->max_value;

        if (!stream.eof()) {
            stream >> this->shuffle_rate;
            if (!stream.eof()) {
                stream >> this->end;
            } else {
                this->end = 0;
            }
        } else {
            this->shuffle_rate = 100;
        }
    } else {
        this->max_value = 1.0;
        this->shuffle_rate = 100;
    }

    if (this->max_value <= 0.0)
        ErrorManager::get_instance()->log_error(
            "Invalid max value for random input generator!");
    if (this->shuffle_rate <= 0)
        ErrorManager::get_instance()->log_error(
            "Invalid shuffle rate for random input generator!");

    this->random_values = (float*) malloc (layer->size * sizeof(float));
}

OneHotRandomInputModule::~OneHotRandomInputModule() {
    free(this->random_values);
}

void OneHotRandomInputModule::feed_input(Buffer *buffer) {
    if (end == 0 or timesteps < end) {
        if ((end == 0 or timesteps < end) and timesteps++ % shuffle_rate == 0) {
            std::cout << "============================ SHUFFLE\n";
            if (end != 0) std::cout << "  *  ";
            shuffle(this->random_values, this->max_value, layer->size);
            float *input = buffer->get_input(this->layer);
            for (int nid = 0 ; nid < this->layer->size; ++nid)
                input[nid] = this->random_values[nid];
            buffer->set_dirty(this->layer);
        }
    } else if (timesteps++ == end) {
        std::cout << "========================================== CLEAR\n";
        float *input = buffer->get_input(this->layer);
        for (int nid = 0 ; nid < this->layer->size; ++nid)
            input[nid] = 0.0;
        buffer->set_dirty(this->layer);
    }
}
