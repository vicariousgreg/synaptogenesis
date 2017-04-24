#include <cstdlib>

#include "io/module/one_hot_cyclic_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#include <sstream>
#include <iostream>

void OneHotCyclicInputModule::cycle() {
    vals[index] = 0.0;
    index = (index + 1) % this->layer->size;
    vals[index] = max_value;
    for (int nid = 0 ; nid < this->layer->size; ++nid) {
        std::cout << vals[nid] << " ";
    }
    std::cout << std::endl;
}

OneHotCyclicInputModule::OneHotCyclicInputModule(Layer *layer, std::string params)
        : Module(layer), timesteps(0), index(layer->size-1) {
    std::stringstream stream(params);
    if (!stream.eof()) {
        stream >> this->max_value;

        if (!stream.eof()) {
            stream >> this->cycle_rate;
            if (!stream.eof()) {
                stream >> this->end;
            } else {
                this->end = 0;
            }
        } else {
            this->cycle_rate = 100;
        }
    } else {
        this->max_value = 1.0;
        this->cycle_rate = 100;
    }

    if (this->max_value <= 0.0)
        ErrorManager::get_instance()->log_error(
            "Invalid max value for random input generator!");
    if (this->cycle_rate <= 0)
        ErrorManager::get_instance()->log_error(
            "Invalid cycle rate for cyclic input generator!");

    this->vals = (float*) calloc (layer->size, sizeof(float));
}

OneHotCyclicInputModule::~OneHotCyclicInputModule() {
    free(this->vals);
}

void OneHotCyclicInputModule::feed_input(Buffer *buffer) {
    if (end == 0 or timesteps < end) {
        if ((end == 0 or timesteps < end) and timesteps++ % cycle_rate == 0) {
            std::cout << "============================ SHUFFLE\n";
            if (end != 0) std::cout << "  *  ";
            this->cycle();
            float *input = buffer->get_input(this->layer);
            for (int nid = 0 ; nid < this->layer->size; ++nid)
                input[nid] = this->vals[nid];
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
