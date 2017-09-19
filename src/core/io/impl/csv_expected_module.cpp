#include "io/impl/csv_expected_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

REGISTER_MODULE(CSVExpectedModule, "csv_expected", EXPECTED);

CSVExpectedModule::CSVExpectedModule(LayerList layers, ModuleConfig *config)
        : CSVReaderModule(layers, config) {
    for (auto layer : layers)
        if (output_types[layer] != FLOAT)
            ErrorManager::get_instance()->log_error(
                "CSVExpectedModule currently only supports FLOAT output type.");
}

void CSVExpectedModule::feed_expected(Buffer *buffer) {
    if (age == 0)
        for (auto layer : layers)
            buffer->set_expected(layer, this->data[curr_row].cast<Output>());
}
