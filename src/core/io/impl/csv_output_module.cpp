#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#include "io/impl/csv_output_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

REGISTER_MODULE(CSVOutputModule, "csv_output");

CSVOutputModule::CSVOutputModule(LayerList layers, ModuleConfig *config)
        : Module(layers) {
    set_io_type(OUTPUT);
}

CSVOutputModule::~CSVOutputModule() { }

void CSVOutputModule::report_output(Buffer *buffer) {
    for (auto layer : layers) {
        Output* output = buffer->get_output(layer);

        for (int row = 0 ; row < layer->rows; ++row) {
            for (int col = 0 ; col < layer->columns; ++col) {
                int index = (row * layer->columns) + col;

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
                if (row != 0 or col != 0) std::cout << ",";
                std::cout << value;
            }
        }
        std::cout << "\n";
    }
}
