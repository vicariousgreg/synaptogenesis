#include "io/impl/callback_module.h"
#include "util/callback_manager.h"

REGISTER_MODULE(CallbackModule, "callback");

CallbackModule::CallbackModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config) {
    enforce_specified_io_type("callback");

    for (auto layer : layers) {
        auto layer_config = config->get_layer(layer);

        // Get Function
        if (not layer_config->has("function"))
            LOG_ERROR("Unspecified callback function for layer "
                + layer->str() + " in CallbackModule!");
        callbacks[layer] = CallbackManager::get_instance()->get_io_callback(
            layer_config->get("function"));

        // Get id
        if (not layer_config->has("id"))
            LOG_ERROR("Unspecified callback id for layer "
            + layer->str() + " in CallbackModule!");
        ids[layer] = layer_config->get_int("id", 0);
    }
}

void CallbackModule::feed_input_impl(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & INPUT) {
            callbacks[layer](ids[layer], layer->size,
                (void*)buffer->get_input(layer).get());
        }
}

void CallbackModule::feed_expected_impl(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & EXPECTED)
            callbacks[layer](ids[layer], layer->size,
                (void*)buffer->get_expected(layer).get());
}

void CallbackModule::report_output_impl(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & OUTPUT)
            callbacks[layer](ids[layer], layer->size,
                (void*)buffer->get_output(layer).get());
}
