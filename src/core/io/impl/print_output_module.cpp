#include <cstdlib>
#include <climits>
#include <iostream>
#include <sstream>

#include "io/impl/print_output_module.h"


REGISTER_MODULE(PrintOutputModule, "print_output");

PrintOutputModule::PrintOutputModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config),
          counter(0) {
    set_io_type(OUTPUT);

    this->history_length = config->get_int("history_length", 1);

    if (this->history_length <= 0 or this->history_length > 8 * sizeof(Output))
        LOG_ERROR(
            "Bad history length parameter for PrintOutputModule!");

    this->maximum = (1 << this->history_length) - 1;
    this->shift = (8 * sizeof(int)) - this->history_length;
}

void PrintOutputModule::report_output_impl(Buffer *buffer) {
    for (auto layer : layers) {
        Output* output = buffer->get_output(layer);

        // Print bar
        for (int col = 0 ; col < layer->columns; ++col) std::cout << "-";
        std::cout << "\n-----  layer: " << layer->name << " -----\n";
        for (int col = 0 ; col < layer->columns; ++col) std::cout << "-";
        std::cout << "\n";

        for (int row = 0 ; row < layer->rows; ++row) {
            for (int col = 0 ; col < layer->columns; ++col) {
                int index = (row * layer->columns) + col;
                // And to remove distant spikes
                // Multiply by constant factor
                // Divide by mask
                //  -> Bins intensity into a constant number of bins

                float fraction;
                Output out_value = output[index];
                switch (output_types[layer]) {
                    case FLOAT:
                        fraction = out_value.f;
                        break;
                    case INT:
                        fraction = (float)out_value.i / INT_MAX;
                        break;
                    case BIT:
                        fraction = float(out_value.i >> this->shift) / this->maximum;
                        break;
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
}
