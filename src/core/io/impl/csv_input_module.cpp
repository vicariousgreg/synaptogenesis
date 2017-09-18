#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#include "io/impl/csv_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

REGISTER_MODULE(CSVInputModule, "csv_input", INPUT);

CSVInputModule::CSVInputModule(Layer *layer, ModuleConfig *config)
        : Module(layer) {
    std::string filename = config->get_property("filename", "");
    int offset = std::stoi(config->get_property("offset", "0"));
    float normalization = std::stof(config->get_property("normalization", "1"));
    this->exposure = std::stoi(config->get_property("exposure", "1"));
    this->age = this->exposure;

    // Check if file exists
    if (filename == "" or not std::ifstream(filename.c_str()).good())
        ErrorManager::get_instance()->log_error(
            "Could not open CSV file!");

    if (offset < 0)
        ErrorManager::get_instance()->log_error(
            "Bad offset in CSV input module!");

    if (exposure < 1)
        ErrorManager::get_instance()->log_error(
            "Bad exposure length in CSV input module!");

    CsvParser *csvparser = CsvParser_new(filename.c_str(), ",", 0);
    CsvRow *row;

    while ((row = CsvParser_getRow(csvparser)) ) {
        data.push_back(Pointer<float>(layer->size));
        const char **rowFields = CsvParser_getFields(row);
        if (layer->size > CsvParser_getNumFields(row) - offset)
            ErrorManager::get_instance()->log_error("Bad CSV file!");

        float *ptr = data[data.size()-1].get();
        for (int i = 0 ; i < layer->size ; i++) {
            std::stringstream(rowFields[i+offset]) >> ptr[i];
            ptr[i] /= normalization;
        }
        CsvParser_destroy_row(row);
    }
    CsvParser_destroy(csvparser);
    curr_row = 0;
}

CSVInputModule::~CSVInputModule() {
    for (auto pointer : data) pointer.free();
}

void CSVInputModule::feed_input(Buffer *buffer) {
    if (curr_row >= this->data.size()) curr_row = 0;
    if (++age >= exposure) {
        buffer->set_input(this->layer, this->data[curr_row++]);
        this->age = 0;
    }
}
