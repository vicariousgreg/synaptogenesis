#include <cstdlib>
#include <string>
#include <math.h>
#include <cfloat>

#include "io/impl/csv_evaluator_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

REGISTER_MODULE(CSVEvaluatorModule, "csv_evaluator", EXPECTED | OUTPUT);

CSVEvaluatorModule::CSVEvaluatorModule(Layer *layer, ModuleConfig *config)
        : CSVExpectedModule(layer, config) {
    this->correct = 0;
    this->total_SSE = 0;
}

void CSVEvaluatorModule::report_output(Buffer *buffer) {
    Output* output = buffer->get_output(this->layer);
    float max_output = FLT_MIN;
    int max_output_index = 0;
    float SSE = 0.0;

    // Get expected one row back since the expected module will increment
    //   before this function is called during an iteration
    Output* expected = (Output*)this->data[curr_row].get();
    float max_expected = FLT_MIN;
    int max_expected_index = 0;

    for (int row = 0 ; row < this->layer->rows; ++row) {
        for (int col = 0 ; col < this->layer->columns; ++col) {
            int index = (row * this->layer->columns) + col;

            float expect = expected[index].f;
            float value;
            Output out_value = output[index];
            switch (output_type) {
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
    this->correct += (max_output_index == max_expected_index);
    this->total_SSE += SSE;

    // If we hit the end of the CSV file, print stats and reset
    if (this->curr_row == this->data.size() - 1) {
        printf("Correct: %9d / %9d [ %9.6f%% ]    SSE: %f\n",
            correct, this->data.size(),
            100.0 * float(correct) / this->data.size(),
            total_SSE);
        this->correct = 0;
        this->total_SSE = 0;
    }
}
