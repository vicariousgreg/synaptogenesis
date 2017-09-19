#include "io/impl/dsst_input_module.h"
#include "util/error_manager.h"

REGISTER_MODULE(DSSTInputModule, "dsst_input", INPUT);

DSSTInputModule::DSSTInputModule(LayerList layers, ModuleConfig *config)
        : Module(layers) {
    enforce_equal_layer_sizes("dsst_input");

    dsst = DSST::get_instance(true);
    for (auto layer : layers) {
        params[layer] = config->get_property("params", "");
        if (not dsst->add_input_layer(layer, params[layer]))
            ErrorManager::get_instance()->log_error(
                "Failed to add layer to DSST!");
    }
}

void DSSTInputModule::feed_input(Buffer *buffer) {
    for (auto layer : layers) {
        if (dsst->is_dirty(params[layer])) {
            Pointer<float> input = dsst->get_input(params[layer]);
            buffer->set_input(layer, input);
        }
    }
}
