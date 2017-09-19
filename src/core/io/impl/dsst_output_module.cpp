#include <algorithm>
#include <iostream>

#include "io/impl/dsst_output_module.h"
#include "util/error_manager.h"
#include "util/tools.h"

REGISTER_MODULE(DSSTOutputModule, "dsst_output", OUTPUT);

DSSTOutputModule::DSSTOutputModule(LayerList layers, ModuleConfig *config)
        : Module(layers) {
    dsst = DSST::get_instance(true);
    for (auto layer : layers)
        if (not dsst->add_output_layer(layer,
                config->get_property("params", "")))
            ErrorManager::get_instance()->log_error(
                "Failed to add layer to DSST!");
}

void DSSTOutputModule::report_output(Buffer *buffer) {
    for (auto layer : layers) {
        Output* output = buffer->get_output(layer);
    }
}
