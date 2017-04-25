#include <cstdlib>

#include "io/module/one_hot_cyclic_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#include <sstream>
#include <iostream>

void OneHotCyclicInputModule::cycle() {
    index = (index + 1) % this->layer->size;
    for (int nid = 0 ; nid < this->layer->size; ++nid) {
        if (nid == index) {
            std::cout << max_value << " ";
        } else {
            std::cout << 0 << " ";
        }
    }
    std::cout << std::endl;
}

void OneHotCyclicInputModule::update(Buffer *buffer) {
    float *input = buffer->get_input(this->layer);
    for (int nid = 0 ; nid < this->layer->size; ++nid) {
        float new_val = input[nid];
        if (nid == index) {
            new_val += (max_value - new_val) / 10;
            if (new_val > 0.99 * max_value) new_val = max_value;
        } else {
            new_val -= new_val / 10;
            if (new_val < 0.01) new_val = 0.0;
        }
        input[nid] = new_val;
    }
    buffer->set_dirty(this->layer);
}

void OneHotCyclicInputModule::clear(Buffer *buffer) {
    float *input = buffer->get_input(this->layer);
    for (int nid = 0 ; nid < this->layer->size; ++nid)
        input[nid] = 0.0;
    buffer->set_dirty(this->layer);
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
}

void OneHotCyclicInputModule::feed_input(Buffer *buffer) {
    if (end == 0 or timesteps <= end) {
        // Cycle if necessary
        if (timesteps % cycle_rate == 0) {
            std::cout << "============================ SHUFFLE\n";
            if (end != 0) std::cout << "  *  ";
            this->cycle();
        }

        // Clear or update
        if (end != 0 and timesteps == end) {
            std::cout << "========================================== CLEAR\n";
            this->clear(buffer);
        } else {
            this->update(buffer);
        }
        ++timesteps;
    }
}
