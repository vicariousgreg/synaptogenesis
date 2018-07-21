#include <string>
#include <cfloat>
#include <iostream>
#include <sstream>
#include <fstream>

#include "io/impl/csv_module.h"
#include "csvparser.h"


REGISTER_MODULE(CSVReaderModule, "csv_reader");
REGISTER_MODULE(CSVExpectedModule, "csv_expected");
REGISTER_MODULE(CSVOutputModule, "csv_output");
REGISTER_MODULE(CSVEvaluatorModule, "csv_evaluator");

/******************************************************************************/
/***************************** CSV READER *************************************/
/******************************************************************************/

CSVReaderModule::CSVReaderModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config) {
    enforce_equal_layer_sizes("csv_reader");

    this->filename = config->get("filename", "");
    int offset = config->get_int("offset", 0);
    float normalization = config->get_float("normalization", 1);
    this->exposure = config->get_int("exposure", 1);
    this->epochs = config->get_int("epochs", 1);

    LOG_DEBUG("Opening file: " + this->filename + " in layers:\n");
    for (auto layer : layers)
        LOG_DEBUG("  " + layer->str());

    // Check if file exists
    if (filename == "" or not std::ifstream(filename.c_str()).good())
        LOG_ERROR(
            "Could not open CSV file!");

    if (offset < 0)
        LOG_ERROR(
            "Bad offset in CSV input module!");

    if (exposure < 1)
        LOG_ERROR(
            "Bad exposure length in CSV input module!");

    CsvParser *csvparser = CsvParser_new(filename.c_str(), ",", 0);
    CsvRow *row;

    std::vector<CsvRow*> rows;
    while ((row = CsvParser_getRow(csvparser)) )
        rows.push_back(row);

    this->num_rows = rows.size();
    int layer_size = layers.at(0)->size;
    this->data = Pointer<float>(layer_size * num_rows);

    int pointer_offset = 0;
    for (auto row : rows) {
        Pointer<float> pointer = data.slice(pointer_offset, layer_size);
        pointers.push_back(pointer);
        pointer_offset += layer_size;

        const char **rowFields = CsvParser_getFields(row);
        if (layer_size > CsvParser_getNumFields(row) - offset)
            LOG_ERROR("Bad CSV file!");

        float *ptr = pointer.get();
        for (int i = 0 ; i < layer_size ; i++)
            ptr[i] = std::atof(rowFields[i+offset]) / normalization;
        CsvParser_destroy_row(row);
    }
    CsvParser_destroy(csvparser);
    curr_row = 0;
}

CSVReaderModule::~CSVReaderModule() {
    data.free();
}

void CSVReaderModule::cycle_impl() {
    if ((curr_iteration % exposure == 0)
            and (++curr_row >= this->num_rows))
        curr_row = 0;
}

/******************************************************************************/
/****************************** CSV INPUT *************************************/
/******************************************************************************/

REGISTER_MODULE(CSVInputModule, "csv_input");

CSVInputModule::CSVInputModule(LayerList layers, ModuleConfig *config)
        : CSVReaderModule(layers, config) {
    set_io_type(INPUT);
}

void CSVInputModule::feed_input_impl(Buffer *buffer) {
    if (curr_iteration % exposure == 0)
        for (auto layer : layers)
            buffer->set_input(layer, this->pointers[curr_row]);
}

/******************************************************************************/
/**************************** CSV EXPECTED ************************************/
/******************************************************************************/

CSVExpectedModule::CSVExpectedModule(LayerList layers, ModuleConfig *config)
        : CSVReaderModule(layers, config) {
    set_io_type(INPUT);
    add_input_auxiliary_key("expected");
    for (auto layer : layers) {
        if (output_types[layer] != FLOAT)
            LOG_ERROR(
                "CSVExpectedModule currently only supports FLOAT output type.");
    }
}

void CSVExpectedModule::feed_input_impl(Buffer *buffer) {
    if (curr_iteration % exposure == 0)
        for (auto layer : layers) {
            auto exp = Pointer<float>(
                buffer->get_input_auxiliary(layer, "expected"));
            this->pointers[curr_row].copy_to(exp);
        }
}

/******************************************************************************/
/****************************** CSV OUTPUT ************************************/
/******************************************************************************/

CSVOutputModule::CSVOutputModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config) {
    set_io_type(OUTPUT);
}

CSVOutputModule::~CSVOutputModule() { }

void CSVOutputModule::report_output_impl(Buffer *buffer) {
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
    set_io_type(INPUT | OUTPUT);
    for (auto layer : layers) {
        this->correct[layer] = 0;
        this->total_SSE[layer] = 0;
    }
}

void CSVEvaluatorModule::report_output_impl(Buffer *buffer) {
    for (auto layer : layers) {
        Output* output = buffer->get_output(layer);
        float max_output = FLT_MIN;
        int max_output_index = 0;
        float SSE = 0.0;

        Output* expected = (Output*)this->pointers[curr_row].get();
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

        // If we hit the end of the CSV file, print stats
        if (verbose and this->curr_row == this->num_rows - 1)
            printf("Correct: %9d / %9d [ %9.6f%% ]    SSE: %f\n",
                corr, this->num_rows,
                100.0 * float(corr) / this->num_rows,
                sse);
    }
}

void CSVEvaluatorModule::report(Report *report) {
    for (auto layer : layers) {
        auto corr = correct[layer];
        auto percentage = 100.0 * corr / num_rows;
        report->add_report(this, layer,
            PropertyConfig({
                { "Filename", std::string(this->filename) },
                { "Samples", std::to_string(num_rows) },
                { "Correct", std::to_string(corr) },
                { "Percentage", std::to_string(percentage) },
                { "SSE", std::to_string(total_SSE[layer]) }
            }));
    }
}
