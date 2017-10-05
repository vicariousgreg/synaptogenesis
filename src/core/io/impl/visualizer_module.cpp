#include "visualizer_module.h"

/******************************************************************************/
/***************************** VISUALIZER *************************************/
/******************************************************************************/

REGISTER_MODULE(VisualizerModule, "visualizer");

VisualizerModule::VisualizerModule(LayerList layers, ModuleConfig *config)
    : Module(layers), window(VisualizerWindow::build_visualizer()) {
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
            set_io_type(layer, OUTPUT);
    }
}

void VisualizerModule::feed_input(Buffer *buffer) {
    for (auto layer : layers) {
        if (get_io_type(layer) & INPUT) {
            buffer->set_dirty(layer);
            window->feed_input(layer, buffer->get_input(layer));
        }
    }
}

void VisualizerModule::report_output(Buffer *buffer) {
    for (auto layer : layers) {
        if (get_io_type(layer) & OUTPUT) {
            Output *output = buffer->get_output(layer);
            OutputType output_type = get_output_type(layer);
            window->report_output(layer, output, output_type);
        }
    }
}

/******************************************************************************/
/******************************* HEATMAP **************************************/
/******************************************************************************/

REGISTER_MODULE(HeatmapModule, "heatmap");

HeatmapModule::HeatmapModule(LayerList layers, ModuleConfig *config)
    : Module(layers), window(VisualizerWindow::build_heatmap()) {
    for (auto layer : layers) {
        auto params =
            config->get_layer(layer)->get("params", "output");
        if (params == "input") {
            set_io_type(layer, INPUT);
            window->add_layer(layer, INPUT);
        } else if (params == "output") {
            set_io_type(layer, OUTPUT);
            window->add_layer(layer, OUTPUT);
        } else {
            LOG_ERROR(
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
