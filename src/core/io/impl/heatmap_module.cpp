#include "heatmap_module.h"

REGISTER_MODULE(HeatmapModule, "heatmap");

HeatmapModule::HeatmapModule(LayerList layers, ModuleConfig *config)
    : Module(layers), window(VisualizerWindow::build_heatmap()) {
    for (auto layer : layers) {
        auto params =
            config->get_layer(layer)->get_property("params", "output");
        if (params == "input") {
            set_io_type(layer, INPUT);
            window->add_layer(layer, INPUT);
        } else if (params == "output") {
            set_io_type(layer, OUTPUT);
            window->add_layer(layer, OUTPUT);
        } else {
            ErrorManager::get_instance()->log_error(
                "Unrecognized layer type: " + params
                + " in HeatmapModule!");
        }
    }
}

void HeatmapModule::report_output(Buffer *buffer) {
    for (auto layer : layers) {
        if (get_io_type(layer) & OUTPUT) {
            Output *output = buffer->get_output(layer);
            OutputType output_type = get_output_type(layer);
            window->report_output(layer, output, output_type);
        }
    }
}
