#include "visualizer_window.h"
#include "impl/visualizer_window_impl.h"

VisualizerWindow* VisualizerWindow::build_visualizer(PropertyConfig *config) {
    return new VisualizerWindowImpl(config);
}

VisualizerWindow* VisualizerWindow::build_heatmap(PropertyConfig *config) {
    return new HeatmapWindowImpl(config);
}
