#include <cstdlib>
#include <climits>
#include <iostream>

#include "io/impl/print_rate_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

REGISTER_MODULE(PrintRateModule, "print_rate", OUTPUT);

PrintRateModule::PrintRateModule(Layer *layer, ModuleConfig *config)
        : Module(layer),
          timesteps(0) {
    this->rate = std::stoi(config->get_property("rate", "100"));

    if (this->rate <= 0)
        ErrorManager::get_instance()->log_error(
            "Invalid rate for print rate module!");

    this->totals = (float*) malloc(layer->size * sizeof(float));
    for (int index = 0 ; index < this->layer->size; ++index)
        totals[index] = 0.0;
}

void PrintRateModule::report_output(Buffer *buffer) {
    Output* output = buffer->get_output(this->layer);

    // Print bar
    /*
    for (int col = 0 ; col < this->layer->columns; ++col) std::cout << "-";
    std::cout << "\n-----  layer: " << this->layer->name << " -----\n";
    for (int col = 0 ; col < this->layer->columns; ++col) std::cout << "-";
    std::cout << "\n";
    */

    for (int row = 0 ; row < this->layer->rows; ++row) {
        for (int col = 0 ; col < this->layer->columns; ++col) {
            int index = (row * this->layer->columns) + col;
            // And to remove distant spikes
            // Multiply by constant factor
            // Divide by mask
            //  -> Bins intensity into a constant number of bins

            Output out_value = output[index];
            switch (output_type) {
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
            if ((timesteps+1) % this->rate == 0) {
                printf("%d\n", (unsigned int) this->totals[index]);
                this->totals[index] = 0.0;
            }
        }
        //std::cout << "|\n";
    }
}

void PrintRateModule::cycle() {
    ++timesteps;
}
