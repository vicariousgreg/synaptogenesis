#ifndef visualizer_window_h
#define visualizer_window_h

#include "network/layer.h"
#include "util/constants.h"

class VisualizerWindow {
    public:
        static VisualizerWindow* build_visualizer();
        static VisualizerWindow* build_heatmap();
        virtual void add_layer(Layer *layer, IOTypeMask io_type) = 0;
        virtual void feed_input(Layer *layer, float *input) = 0;
        virtual void report_output(Layer *layer,
            Output *output, OutputType output_type) = 0;
};

#endif
