#include "io/impl/dsst_module.h"
#include "util/error_manager.h"
#include "dsst_window.h"

REGISTER_MODULE(DSSTModule, "dsst");

DSSTModule::DSSTModule(LayerList layers, ModuleConfig *config)
        : Module(layers),
          window(DSSTWindow::build()) {
    for (auto layer : layers) {
        auto param =
            config->get_layer(layer)->get_property("params", "output");
        params[layer] = param;

        if (param == "input") {
            set_io_type(layer, INPUT);
            window->add_layer(layer, INPUT);
        } else if (param == "output") {
            set_io_type(layer, OUTPUT);
            window->add_layer(layer, OUTPUT);
        } else {
            ErrorManager::get_instance()->log_error(
                "Unrecognized layer type: " + param
                + " in VisualizerModule!");
        }
        set_io_type(INPUT);
    }
    input_data = Pointer<float>(DSSTModule::input_size);
}

void DSSTModule::feed_input(Buffer *buffer) {
    for (auto layer : layers) {
        if (get_io_type(layer) & INPUT) {
            window->feed_input(layer, buffer->get_input(layer));
            buffer->set_dirty(layer, true);
        }
    }
}

void DSSTModule::report_output(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & OUTPUT) ;
            // Add output processing here
}

void DSSTModule::input_symbol(int index) {
    window->input_symbol(index);
}
