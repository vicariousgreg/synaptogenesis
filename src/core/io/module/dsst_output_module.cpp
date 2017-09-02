#include <algorithm>

#include "io/module/dsst_output_module.h"
#include "util/error_manager.h"

REGISTER_MODULE(DSSTOutputModule, "dsst_output", OUTPUT);

DSSTOutputModule::DSSTOutputModule(Layer *layer, ModuleConfig *config)
        : Module(layer) {
    dsst = DSST::get_instance(true);
    if (not dsst->add_output_layer(layer, config->get_property("params")))
        ErrorManager::get_instance()->log_error(
            "Failed to add layer to DSST!");
}

void DSSTOutputModule::report_output(Buffer *buffer, OutputType output_type) {
    Output* output = buffer->get_output(this->layer);
}
