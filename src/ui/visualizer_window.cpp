#include "visualizer_window.h"
#include "impl/visualizer_window_impl.h"

VisualizerWindow* VisualizerWindow::build_visualizer() {
    return new VisualizerWindowImpl();
}

VisualizerWindow* VisualizerWindow::build_heatmap(int rate, bool linear) {
    if (rate < 1)
        LOG_ERROR("Invalid rate in HeatmapModule: " + std::to_string(rate));

    return new HeatmapWindowImpl(rate, linear);
}
