#include <cstdlib>
#include <string>
#include <math.h>
#include <cfloat>

#include "io/impl/csv_evaluator_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

REGISTER_MODULE(CSVEvaluatorModule, "csv_evaluator", EXPECTED | OUTPUT);

CSVEvaluatorModule::CSVEvaluatorModule(LayerList layers, ModuleConfig *config)
        : CSVExpectedModule(layers, config) {
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
