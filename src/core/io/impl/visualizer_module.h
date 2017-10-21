#ifndef visualizer_module_h
#define visualizer_module_h

#include "io/module.h"

#include "visualizer_window.h"

class VisualizerModule : public Module {
    public:
        VisualizerModule(LayerList layers, ModuleConfig *config);

        void feed_input_impl(Buffer *buffer);
        void report_output_impl(Buffer *buffer);

    protected:
        VisualizerWindow *window;

    MODULE_MEMBERS
};

class HeatmapModule : public Module {
    public:
        HeatmapModule(LayerList layers, ModuleConfig *config);

        void report_output_impl(Buffer *buffer);
        void cycle_impl();

    protected:
        VisualizerWindow *window;

    MODULE_MEMBERS
};

#endif
