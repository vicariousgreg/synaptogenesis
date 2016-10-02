#include <cstdlib>
#include <iostream>

#include "spike_print_output.h"
#include "../framework/tools.h"

SpikePrintOutput::SpikePrintOutput(Layer *layer, std::string params)
        : Output(layer) { }

void SpikePrintOutput::report_output(State *state) {
    int* spikes = (int*)state->get_output();
    for (int row = 0 ; row < this->layer->rows; ++row) {
        for (int col = 0 ; col < this->layer->columns; ++col) {
            int index = (row * this->layer->columns) + col;
            std::cout << ((spikes[index+layer->index] % 2) ? "* " : "  ");
        }
        std::cout << "|\n";
    }
    for (int col = 0 ; col < this->layer->columns; ++col) {
        std::cout << "-";
    }
    std::cout << " layer id: " << this->layer->id << " ";
    for (int col = 0 ; col < this->layer->columns; ++col) {
        std::cout << "-";
    }
    std::cout << "\n";
}
