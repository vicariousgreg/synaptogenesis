#include "io/impl/csv_expected_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

REGISTER_MODULE(CSVExpectedModule, "csv_expected", EXPECTED);

CSVExpectedModule::CSVExpectedModule(Layer *layer, ModuleConfig *config)
        : CSVReaderModule(layer, config) {
    if (output_type != FLOAT)
        ErrorManager::get_instance()->log_error(
            "CSVExpectedModule currently only supports FLOAT output type.");
}

void CSVExpectedModule::feed_expected(Buffer *buffer) {
    if (age == 0)
        buffer->set_expected(this->layer, this->data[curr_row].cast<Output>());
}
