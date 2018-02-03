#ifdef __GUI__

#include "io/impl/saccade_module.h"

#include "saccade_window.h"

REGISTER_MODULE(SaccadeModule, "saccade");

SaccadeModule::SaccadeModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config) {
    input_data = Pointer<float>(get_input_size());
    this->window = SaccadeWindow::build(this);

    for (auto layer : layers) {
        auto layer_config = config->get_layer(layer);

        if (layer_config->get_bool("input", false))
            set_io_type(layer, get_io_type(layer) | INPUT);

        if (layer_config->get_bool("expected", false))
            set_io_type(layer, get_io_type(layer) | EXPECTED);

        if (layer_config->get_bool("output", false))
            set_io_type(layer, get_io_type(layer) | OUTPUT);

        // Use output as default
        if (get_io_type(layer) == 0)
            set_io_type(layer, INPUT);
        window->add_layer(layer, get_io_type(layer));
    }
}

void SaccadeModule::feed_input_impl(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & INPUT)
            window->feed_input(layer, buffer->get_input(layer));
}

void SaccadeModule::report_output_impl(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & OUTPUT) ;
            // Add output processing here
}

#endif
