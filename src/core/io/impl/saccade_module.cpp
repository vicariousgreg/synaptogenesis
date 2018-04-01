#ifdef __GUI__

#include "io/impl/saccade_module.h"

#include "saccade_window.h"

REGISTER_MODULE(SaccadeModule, "saccade");

SaccadeModule::SaccadeModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config) {
    input_data = Pointer<float>(get_input_size());
    this->window = SaccadeWindow::build(this);

    // Use input as default
    set_default_io_type(INPUT);

    for (auto layer : layers)
        window->add_layer(layer, get_io_type(layer));
}

void SaccadeModule::feed_input_impl(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & INPUT)
            window->feed_input(layer, buffer->get_input(layer));
}

void SaccadeModule::report_output_impl(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & OUTPUT)
            window->report_output(layer,
                buffer->get_output(layer),
                get_output_type(layer));
}

#endif
