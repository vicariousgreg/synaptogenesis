#include "io/module/impl/dsst_input_module.h"
#include "util/error_manager.h"

REGISTER_MODULE(DSSTInputModule, "dsst_input", INPUT);

DSSTInputModule::DSSTInputModule(Layer *layer, ModuleConfig *config)
        : Module(layer), params(config->get_property("params")) {
    dsst = DSST::get_instance(true);
    if (not dsst->add_input_layer(layer, params))
        ErrorManager::get_instance()->log_error(
            "Failed to add layer to DSST!");
}

void DSSTInputModule::feed_input(Buffer *buffer) {
    if (dsst->is_dirty(params)) {
        Pointer<float> input = dsst->get_input(params);
        buffer->set_input(this->layer, input);
    }
}
