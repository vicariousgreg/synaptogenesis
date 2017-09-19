#include "io/impl/csv_input_module.h"

REGISTER_MODULE(CSVInputModule, "csv_input", INPUT);

CSVInputModule::CSVInputModule(LayerList layers, ModuleConfig *config)
        : CSVReaderModule(layers, config) { }

void CSVInputModule::feed_input(Buffer *buffer) {
    if (age == 0)
        for (auto layer : layers)
            buffer->set_input(layer, this->data[curr_row]);
}
