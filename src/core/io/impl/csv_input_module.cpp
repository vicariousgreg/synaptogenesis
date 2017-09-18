#include "io/impl/csv_input_module.h"

REGISTER_MODULE(CSVInputModule, "csv_input", INPUT);

CSVInputModule::CSVInputModule(Layer *layer, ModuleConfig *config)
        : CSVReaderModule(layer, config) { }

void CSVInputModule::feed_input(Buffer *buffer) {
    if (age == 0)
        buffer->set_input(this->layer, this->data[curr_row]);
}
