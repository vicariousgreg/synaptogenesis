#include <cstdlib>
#include <climits>
#include <iostream>
#include <sstream>

#include "io/module/print_rate_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

PrintRateModule::PrintRateModule(Layer *layer, std::string params)
        : Module(layer),
          timesteps(0) {
    this->totals = (float*) malloc(layer->size * sizeof(float));
    for (int index = 0 ; index < this->layer->size; ++index)
        totals[index] = 0.0;
}

void PrintRateModule::report_output(Buffer *buffer) {
    Output* output = buffer->get_output();
    OutputType output_type = buffer->get_output_type();

    timesteps++;

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

            float fraction;
            Output out_value = output[index+layer->output_index];
            switch (output_type) {
                case FLOAT:
                    fraction = out_value.f;
                    break;
                case INT:
                    fraction = (float)out_value.i / INT_MAX;
                    break;
                case BIT:
                    int total = 0;
                    for (int i = 0; i < 8 * sizeof(int); ++i)
                        total += (out_value.i & (1 << i)) ? 1 : 0;
                    fraction = total;
                    break;
            }
            this->totals[index] += fraction;
            if (timesteps % 50 == 0) {
                printf("%d\n", (unsigned int) this->totals[index]);
                this->totals[index] = 0.0;
            }
        }
        //std::cout << "|\n";
    }
}
