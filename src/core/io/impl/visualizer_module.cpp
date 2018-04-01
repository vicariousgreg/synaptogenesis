#ifdef __GUI__

#include "io/impl/visualizer_module.h"

#include "visualizer_window.h"

/******************************************************************************/
/***************************** VISUALIZER *************************************/
/******************************************************************************/

REGISTER_MODULE(VisualizerModule, "visualizer");

VisualizerModule::VisualizerModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config),
          window(VisualizerWindow::build_visualizer(config)) {
    // Use output as default
    set_default_io_type(OUTPUT);

    for (auto layer : layers)
        window->add_layer(layer, get_io_type(layer));
}

void VisualizerModule::feed_input_impl(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & INPUT)
            window->feed_input(layer, buffer->get_input(layer));
}

void VisualizerModule::report_output_impl(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & OUTPUT)
            window->report_output(layer,
                buffer->get_output(layer),
                get_output_type(layer));
}

/******************************************************************************/
/******************************* HEATMAP **************************************/
/******************************************************************************/

REGISTER_MODULE(HeatmapModule, "heatmap");

HeatmapModule::HeatmapModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config),
          window(VisualizerWindow::build_heatmap(config)) {
    // Use output as default
    set_default_io_type(OUTPUT);

    for (auto layer : layers)
        window->add_layer(layer, get_io_type(layer));
}

void HeatmapModule::report_output_impl(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & OUTPUT)
            window->report_output(layer,
                buffer->get_output(layer),
                get_output_type(layer));
}

void HeatmapModule::cycle_impl() {
    window->cycle();
}

#endif
