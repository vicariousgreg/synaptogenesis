#include <cstdlib>
#include <iostream>
#include <sstream>

#include "io/print_output_module.h"
#include "tools.h"

PrintOutputModule::PrintOutputModule(Layer *layer,
        std::string params, std::string &driver_type)
        : Module(layer, driver_type),
          counter(0) {
    std::stringstream stream(params);
    if (!stream.eof()) {
        stream >> this->history_length;
        if (this->history_length <= 0 or this->history_length > 8 * sizeof(Output))
            throw "Bad history length parameter for PrintOutputModule!";
    } else {
        this->history_length= 1;
    }

    // Set up maximum value
    this->maximum = (1 << this->history_length) - 1;

    // Create table of reversed bit strings
    this->reverses = (unsigned int*)malloc(this->maximum * sizeof(unsigned int));
    for (unsigned int i = 0 ; i < this->maximum ; ++i) {
        unsigned int copy = i;
        unsigned int reverse = 0;
        for (unsigned int j = 0 ; j < history_length ; ++j) {
            reverse <<= 1;
            reverse |= copy % 2;
            copy >>= 1;
        }
        this->reverses[i] = reverse;
    }
}

void PrintOutputModule::report_output(Buffer *buffer) {
    return;
    Output* output = buffer->get_output();

    // Print bar
    for (int col = 0 ; col < this->layer->columns; ++col) std::cout << "-";
    std::cout << "\n-----  layer id: " << this->layer->id << "-----\n";
    for (int col = 0 ; col < this->layer->columns; ++col) std::cout << "-";
    std::cout << "\n";

    for (int row = 0 ; row < this->layer->rows; ++row) {
        for (int col = 0 ; col < this->layer->columns; ++col) {
            int index = (row * this->layer->columns) + col;
            // And to remove distant spikes
            // Multiply by constant factor
            // Divide by mask
            //  -> Bins intensity into a constant number of bins

            float fraction;
            Output out_value = output[index+layer->output_index];
            if (this->driver_type == "izhikevich") {
                unsigned int spike_value = (unsigned int) (out_value.i & this->maximum);
                unsigned int value = this->reverses[spike_value];
                fraction = float(value) / this->maximum;
            } else if (this->driver_type == "rate_encoding") {
                fraction = out_value.f;
            } else {
            }
            if (fraction > 0) {
                //std::cout << value << " ";
                //std::cout << fraction << " ";
                if (fraction > 0.24)      std::cout << " X";
                else if (fraction > 0.06) std::cout << " @";
                else if (fraction > 0.01) std::cout << " +";
                else if (fraction > 0.002) std::cout << " *";
                else if (fraction > 0.001) std::cout << " -";
                else                     std::cout << " '";
            } else {
                std::cout << "  ";
            }
        }
        std::cout << "|\n";
    }
}
