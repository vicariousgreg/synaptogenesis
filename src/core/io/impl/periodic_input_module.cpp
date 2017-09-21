#include "io/impl/periodic_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#include <sstream>
#include <iostream>

REGISTER_MODULE(BasicPeriodicInputModule, "periodic_input");
REGISTER_MODULE(OneHotRandomInputModule, "one_hot_random_input");
REGISTER_MODULE(OneHotCyclicInputModule, "one_hot_cyclic_input");

PeriodicInputModule::PeriodicInputModule(LayerList layers, ModuleConfig *config)
        : Module(layers), timesteps(0), dirty(true) {
    enforce_equal_layer_sizes("periodic_input");
    set_io_type(INPUT);

    this->value = std::stof(config->get("val", "1.0"));
    this->min_value = std::stof(config->get("min", "0.0"));
    this->max_value = std::stof(config->get("max", "1.0"));
    this->rate = std::stoi(config->get("rate", "1"));
    this->end = std::stoi(config->get("end", "0"));
    this->fraction = std::stof(config->get("fraction", "1.0"));
    this->random = config->get("random", "false") == "true";
    this->clear = config->get("clear", "false") == "true";
    this->verbose = config->get("verbose", "false") == "true";

    if (this->min_value > this->max_value)
        ErrorManager::get_instance()->log_error(
            "Invalid min/max value for periodic input generator!");
    if (this->rate <= 0)
        ErrorManager::get_instance()->log_error(
            "Invalid rate for periodic input generator!");
    if (this->fraction <= 0.0 or this->fraction > 1.0)
        ErrorManager::get_instance()->log_error(
            "Invalid fraction for periodic input generator!");
    if (this->end < 0)
        ErrorManager::get_instance()->log_error(
            "Invalid shuffle rate for random input generator!");

    this->values = Pointer<float>(layers.at(0)->size, 0.0);
}

PeriodicInputModule::~PeriodicInputModule() {
    this->values.free();
}

void PeriodicInputModule::feed_input(Buffer *buffer) {
    if (dirty) {
        for (auto layer : layers) {
            this->values.copy_to(buffer->get_input(layer));
            buffer->set_dirty(layer);
        }
        dirty = false;
    }
}

void PeriodicInputModule::cycle() {
    if (end == 0 or timesteps < end) {
        if (++timesteps % rate == 0) {
            dirty = true;
            this->update();

            if (verbose) {
                std::cout << "============================ SHUFFLE\n";
                for (int nid = 0 ; nid < values.get_size(); ++nid)
                    std::cout << this->values[nid] << " ";
                std::cout << std::endl;
            }
        } else if (clear and (timesteps % rate == 1)) {
            dirty = true;
            if (verbose) std::cout << "============================ CLEAR\n";
            fSet(values, values.get_size(), 0.0);
        }
    } else if (timesteps == end) {
        dirty = true;
        if (verbose) std::cout << "============================ CLEAR\n";
        fSet(values, values.get_size(), 0.0);
    }
}

void BasicPeriodicInputModule::update() {
    if (random)
        fRand(values, values.get_size(), min_value, max_value, fraction);
    else
        fSet(values, values.get_size(), value, fraction);
}

void OneHotRandomInputModule::update() {
    /*  Randomly selects one input to activate */
    int random_index = rand() % values.get_size();

    if (random)
        for (int nid = 0 ; nid < values.get_size(); ++nid)
            values[nid] =
                (nid == random_index)
                ? fRand(min_value, max_value) : 0;
    else
        for (int nid = 0 ; nid < values.get_size(); ++nid)
            values[nid] =
                (nid == random_index) ? value : 0;
}

void OneHotCyclicInputModule::update() {
    int index = timesteps % values.get_size();

    if (random)
        for (int nid = 0 ; nid < values.get_size(); ++nid)
            values[nid] =
                (nid == index)
                ? fRand(min_value, max_value) : 0;
    else
        for (int nid = 0 ; nid < values.get_size(); ++nid)
            values[nid] =
                (nid == index) ? value : 0;
}
