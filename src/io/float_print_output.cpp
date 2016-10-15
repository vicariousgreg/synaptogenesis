#include <cstdlib>
#include <iostream>

#include "io/float_print_output.h"
#include "tools.h"

FloatPrintOutput::FloatPrintOutput(Layer *layer, std::string params)
        : Output(layer) { }

void FloatPrintOutput::report_output(Buffer *buffer) {
    float* inputs = (float*) buffer->get_output();
    for (int row = 0 ; row < this->layer->rows; ++row) {
        for (int col = 0 ; col < this->layer->columns; ++col) {
            int index = (row * this->layer->columns) + col;
            std::cout << inputs[index+layer->index] << " ";
        }
        std::cout << "|\n";
    }
    for (int col = 0 ; col < this->layer->columns; ++col) {
        std::cout << "-";
    }
    std::cout << "\n";
}
