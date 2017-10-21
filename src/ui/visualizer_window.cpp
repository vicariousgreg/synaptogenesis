#include "visualizer_window.h"
#include "impl/visualizer_window_impl.h"

VisualizerWindow* VisualizerWindow::build_visualizer() {
    return new VisualizerWindowImpl();
}

VisualizerWindow* VisualizerWindow::build_heatmap(
        int integration_window, bool linear) {
    if (integration_window < 1)
        LOG_ERROR("Invalid integrationwindow in HeatmapModule: "
            + std::to_string(integration_window));

    return new HeatmapWindowImpl(integration_window, linear);
}
