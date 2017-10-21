#include <cstdlib>
#include <climits>
#include <iostream>

#include "io/impl/print_rate_module.h"
#include "util/tools.h"


REGISTER_MODULE(PrintRateModule, "print_rate");

PrintRateModule::PrintRateModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config),
          timesteps(0) {
    set_io_type(OUTPUT);

    this->window = config->get_int("window", 100);

    if (this->window <= 0)
        LOG_ERROR(
            "Invalid window for print rate module!");

    this->totals = (float*) malloc(layers.at(0)->size * sizeof(float));
    for (int index = 0 ; index < this->layers.at(0)->size; ++index)
        totals[index] = 0.0;
}

void PrintRateModule::report_output_impl(Buffer *buffer) {
    for (auto layer : layers) {
        Output* output = buffer->get_output(layer);

        // Print bar
        /*
        for (int col = 0 ; col < layer->columns; ++col) std::cout << "-";
        std::cout << "\n-----  layer: " << layer->name << " -----\n";
        for (int col = 0 ; col < layer->columns; ++col) std::cout << "-";
        std::cout << "\n";
        */

        for (int row = 0 ; row < layer->rows; ++row) {
            for (int col = 0 ; col < layer->columns; ++col) {
                int index = (row * layer->columns) + col;
                // And to remove distant spikes
                // Multiply by constant factor
                // Divide by mask
                //  -> Bins intensity into a constant number of bins

                Output out_value = output[index];
                switch (output_types[layer]) {
                    case FLOAT:
                        totals[index] += out_value.f;
                        break;
                    case INT:
                        totals[index] += (float)out_value.i / INT_MAX;
                        break;
                    case BIT:
                        totals[index] += (out_value.i >> 31);
                        break;
                }
                if ((timesteps+1) % this->window == 0) {
                    printf("%d\n", (unsigned int) this->totals[index]);
                    this->totals[index] = 0.0;
                }
            }
            //std::cout << "|\n";
        }
    }
}

void PrintRateModule::cycle_impl() {
    ++timesteps;
}
