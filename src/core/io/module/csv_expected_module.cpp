#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#include "io/module/csv_expected_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

REGISTER_MODULE(CSVExpectedModule, "csv_expected", INPUT | EXPECTED);

CSVExpectedModule::CSVExpectedModule(Layer *layer, ModuleConfig *config)
        : Module(layer) {
    std::string filename;
    std::stringstream stream(config->get_property("params"));
    stream >> filename;

    // Check if file exists
    if (not std::ifstream(filename.c_str()).good())
        ErrorManager::get_instance()->log_error(
            "Could not open CSV file!");

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

    // Alternate input/expected with CSV lines
    bool input_switch = true;
    while ((row = CsvParser_getRow(csvparser)) ) {
        if (input_switch)
            input.push_back(Pointer<float>(layer->size));
        else
            expected.push_back(Pointer<Output>(layer->size));
        const char **rowFields = CsvParser_getFields(row);
        if (layer->size > CsvParser_getNumFields(row))
            ErrorManager::get_instance()->log_error("Bad CSV file!");

        float *ptr;
        if (input_switch)
            ptr = (float*)input[input.size()-1].get();
        else
            ptr = (float*)expected[input.size()-1].get();
        for (int i = 0 ; i < layer->size ; i++) {
            std::stringstream(rowFields[i]) >> ptr[i];
            ptr[i] /= normalization;
        }
        CsvParser_destroy_row(row);

        input_switch = not input_switch;
    }
    CsvParser_destroy(csvparser);
    curr_row = 0;
}

CSVExpectedModule::~CSVExpectedModule() {
    for (auto pointer : input) delete pointer;
}

void CSVExpectedModule::feed_input(Buffer *buffer) {
    if (++age > exposure and curr_row <= this->input.size()) {
        buffer->set_input(this->layer, this->input[curr_row++]);
        this->age = 0;
    }
}
void CSVExpectedModule::feed_expected(Buffer *buffer) {
    if (++age > exposure and curr_row <= this->expected.size()) {
        buffer->set_expected(this->layer, this->expected[curr_row++]);
        this->age = 0;
    }
}
