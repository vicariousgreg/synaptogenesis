#include "visualizer_window.h"
#include "impl/visualizer_window_impl.h"

VisualizerWindow* VisualizerWindow::build_visualizer() {
    return new VisualizerWindowImpl();
}

VisualizerWindow* VisualizerWindow::build_heatmap() {
    return new HeatmapWindowImpl();
}
