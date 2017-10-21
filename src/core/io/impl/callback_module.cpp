#include "callback_module.h"

REGISTER_MODULE(CallbackModule, "callback");

CallbackModule::CallbackModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config) {
    for (auto layer : layers) {
        auto layer_config = config->get_layer(layer);

        // Get Function
        if (not layer_config->has("function"))
            LOG_ERROR("Unspecified callback function for layer "
            + layer->str() + " in CallbackModule!");
        callbacks[layer] = (void(*)(int, int, void*))(
            std::stoll(layer_config->get("function")));

        // Get id
        if (not layer_config->has("id"))
            LOG_ERROR("Unspecified callback id for layer "
            + layer->str() + " in CallbackModule!");
        ids[layer] = layer_config->get_int("arg", 0);

        if (layer_config->get_bool("input", false))
            set_io_type(layer, get_io_type(layer) | INPUT);

        if (layer_config->get_bool("expected", false))
            set_io_type(layer, get_io_type(layer) | EXPECTED);

        if (layer_config->get_bool("output", false))
            set_io_type(layer, get_io_type(layer) | OUTPUT);

        // Use output as default
        if (get_io_type(layer) == 0)
            set_io_type(layer, OUTPUT);
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
