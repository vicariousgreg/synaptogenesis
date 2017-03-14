#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#include "io/module/csv_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

CSVInputModule::CSVInputModule(Layer *layer, std::string params)
        : Module(layer) {
    std::string filename;
    std::stringstream stream(params);
    stream >> filename;

    // Check if file exists
    if (not std::ifstream(filename.c_str()).good())
        ErrorManager::get_instance()->log_error(
            "Could not open CSV file!");

    // Pull offset
    int offset = 0;
    if (not stream.eof())
        stream >> offset;

    // Pull exposure
    if (not stream.eof())
        stream >> this->exposure;
    else this->exposure = 0;
    this->age = this->exposure;

    // Pull normalization constant
    float normalization = 1;
    if (not stream.eof())
        stream >> normalization;

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
    for (auto pointer : data) delete pointer;
}

void CSVInputModule::feed_input(Buffer *buffer) {
    if (++age > exposure and curr_row <= this->data.size()) {
        buffer->set_input(this->layer, this->data[curr_row++]);
        this->age = 0;
    }
}
