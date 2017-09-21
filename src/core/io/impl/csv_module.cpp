#include <string>
#include <cfloat>
#include <iostream>
#include <sstream>
#include <fstream>

#include "io/impl/csv_module.h"
#include "util/tools.h"
#include "util/error_manager.h"
#include "csvparser.h"


REGISTER_MODULE(CSVReaderModule, "csv_reader");
REGISTER_MODULE(CSVExpectedModule, "csv_expected");
REGISTER_MODULE(CSVOutputModule, "csv_output");
REGISTER_MODULE(CSVEvaluatorModule, "csv_evaluator");

/******************************************************************************/
/***************************** CSV READER *************************************/
/******************************************************************************/

CSVReaderModule::CSVReaderModule(LayerList layers, ModuleConfig *config)
        : Module(layers) {
    enforce_equal_layer_sizes("csv_reader");

    std::string filename = config->get("filename", "");
    int offset = std::stoi(config->get("offset", "0"));
    float normalization = std::stof(config->get("normalization", "1"));
    this->exposure = std::stoi(config->get("exposure", "1"));
    this->age = 0;

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
        data.push_back(Pointer<float>(layers.at(0)->size));
        const char **rowFields = CsvParser_getFields(row);
        if (layers.at(0)->size > CsvParser_getNumFields(row) - offset)
            ErrorManager::get_instance()->log_error("Bad CSV file!");

        float *ptr = data[data.size()-1].get();
        for (int i = 0 ; i < layers.at(0)->size ; i++) {
            std::stringstream(rowFields[i+offset]) >> ptr[i];
            ptr[i] /= normalization;
        }
        CsvParser_destroy_row(row);
    }
    CsvParser_destroy(csvparser);
    curr_row = 0;
}

CSVReaderModule::~CSVReaderModule() {
    for (auto pointer : data) pointer.free();
}

void CSVReaderModule::cycle() {
    if (++age >= exposure) {
        this->age = 0;
        if (++curr_row >= this->data.size()) curr_row = 0;
    }
}

/******************************************************************************/
/****************************** CSV INPUT *************************************/
/******************************************************************************/

REGISTER_MODULE(CSVInputModule, "csv_input");

CSVInputModule::CSVInputModule(LayerList layers, ModuleConfig *config)
        : CSVReaderModule(layers, config) {
    set_io_type(INPUT);
}

void CSVInputModule::feed_input(Buffer *buffer) {
    if (age == 0)
        for (auto layer : layers)
            buffer->set_input(layer, this->data[curr_row]);
}

/******************************************************************************/
/**************************** CSV EXPECTED ************************************/
/******************************************************************************/

CSVExpectedModule::CSVExpectedModule(LayerList layers, ModuleConfig *config)
        : CSVReaderModule(layers, config) {
    set_io_type(EXPECTED);
    for (auto layer : layers)
        if (output_types[layer] != FLOAT)
            ErrorManager::get_instance()->log_error(
                "CSVExpectedModule currently only supports FLOAT output type.");
}

void CSVExpectedModule::feed_expected(Buffer *buffer) {
    if (age == 0)
        for (auto layer : layers)
            buffer->set_expected(layer, this->data[curr_row].cast<Output>());
}

/******************************************************************************/
/****************************** CSV OUTPUT ************************************/
/******************************************************************************/

CSVOutputModule::CSVOutputModule(LayerList layers, ModuleConfig *config)
        : Module(layers) {
    set_io_type(OUTPUT);
}

CSVOutputModule::~CSVOutputModule() { }

void CSVOutputModule::report_output(Buffer *buffer) {
    for (auto layer : layers) {
        Output* output = buffer->get_output(layer);

        for (int row = 0 ; row < layer->rows; ++row) {
            for (int col = 0 ; col < layer->columns; ++col) {
                int index = (row * layer->columns) + col;

                float value;
                Output out_value = output[index];
                switch (output_types[layer]) {
                    case FLOAT:
                        value = out_value.f;
                        break;
                    case INT:
                        value = out_value.i;
                        break;
                    case BIT:
                        value = out_value.i >> 31;
                        break;
                }
                if (row != 0 or col != 0) std::cout << ",";
                std::cout << value;
            }
        }
        std::cout << "\n";
    }
}

/******************************************************************************/
/***************************** CSV EVALUATOR **********************************/
/******************************************************************************/

CSVEvaluatorModule::CSVEvaluatorModule(LayerList layers, ModuleConfig *config)
        : CSVExpectedModule(layers, config) {
    set_io_type(EXPECTED | OUTPUT);
    for (auto layer : layers) {
        this->correct[layer] = 0;
        this->total_SSE[layer] = 0;
    }
}

void CSVEvaluatorModule::report_output(Buffer *buffer) {
    for (auto layer : layers) {
        Output* output = buffer->get_output(layer);
        float max_output = FLT_MIN;
        int max_output_index = 0;
        float SSE = 0.0;

        // Get expected one row back since the expected module will increment
        //   before this function is called during an iteration
        Output* expected = (Output*)this->data[curr_row].get();
        float max_expected = FLT_MIN;
        int max_expected_index = 0;

        for (int row = 0 ; row < layer->rows; ++row) {
            for (int col = 0 ; col < layer->columns; ++col) {
                int index = (row * layer->columns) + col;

                float expect = expected[index].f;
                float value;
                Output out_value = output[index];
                switch (output_types[layer]) {
                    case FLOAT:
                        value = out_value.f;
                        break;
                    case INT:
                        value = out_value.i;
                        break;
                    case BIT:
                        value = out_value.i >> 31;
                        break;
                }
                SSE += pow(value - expect, 2);
                if (value > max_output) {
                    max_output = value;
                    max_output_index = index;
                }
                if (expect > max_expected) {
                    max_expected = expect;
                    max_expected_index = index;
                }
            }
        }
        int corr = (correct[layer] += (max_output_index == max_expected_index));
        float sse = (total_SSE[layer] += SSE);

        // If we hit the end of the CSV file, print stats and reset
        if (this->curr_row == this->data.size() - 1) {
            printf("Correct: %9d / %9d [ %9.6f%% ]    SSE: %f\n",
                corr, this->data.size(),
                100.0 * float(corr) / this->data.size(),
                sse);
            correct[layer] = 0;
            total_SSE[layer] = 0;
        }
    }
}
