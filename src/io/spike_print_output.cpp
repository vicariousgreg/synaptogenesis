#include <cstdlib>
#include <iostream>
#include <sstream>

#include "io/spike_print_output.h"
#include "tools.h"

SpikePrintOutput::SpikePrintOutput(Layer *layer, std::string params)
        : Output(layer),
          counter(0) {
    std::stringstream stream(params);
    if (!stream.eof()) {
        stream >> this->history_length;
        if (!stream.eof())
            stream >> this->step_size;
            if (!stream.eof())
                stream >> this->refresh_rate;
            else
                throw "Insufficient parameters for spike output printer";
    } else {
        this->history_length= 1;
        this->step_size = 1;
        this->refresh_rate = 10;
    }
}

void SpikePrintOutput::report_output(State *state) {
    float time = (this->counter == 0) ? (1.0 / this->refresh_rate): this->timer.query(NULL);
    if (this->counter++ % this->step_size == 0) {
        while (time < (1.0 / this->refresh_rate)) time = this->timer.query(NULL);
        this->timer.start();
        int* spikes = (int*)state->get_output();

        // Print bar
        for (int col = 0 ; col < this->layer->columns; ++col) std::cout << "-";
        std::cout << "\n-----  layer id: " << this->layer->id << "-----\n";
        for (int col = 0 ; col < this->layer->columns; ++col) std::cout << "-";
        std::cout << "\n";

        // Create mask
        int mask = 1;
        if (this->history_length > 1)
            mask = (mask << this->history_length) - 1;

        for (int row = 0 ; row < this->layer->rows; ++row) {
            for (int col = 0 ; col < this->layer->columns; ++col) {
                int index = (row * this->layer->columns) + col;
                // And to remove distant spikes
                // Multiply by constant factor
                // Divide by mask
                //  -> Bins intensity into a constant number of bins

                int value = spikes[index+layer->index] & mask;
                float fraction = float(value) / mask;
                if (value > 0) {
                    //std::cout << value << " ";
                    //std::cout << fraction << " ";
                    if (fraction < 0.01)      std::cout << " '";
                    else if (fraction < 0.02) std::cout << " -";
                    else if (fraction < 0.03) std::cout << " *";
                    else if (fraction < 0.06) std::cout << " +";
                    else if (fraction < 0.2) std::cout << " @";
                    else                     std::cout << " X";
                } else {
                    std::cout << "  ";
                }
            }
            std::cout << "|\n";
        }
    } else {
        this->timer.start();
    }
}
