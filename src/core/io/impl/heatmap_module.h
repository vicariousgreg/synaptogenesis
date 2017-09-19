#ifndef heatmap_module_h
#define heatmap_module_h

#include "io/module.h"
#include "util/error_manager.h"
#include "visualizer_window.h"

class HeatmapModule : public Module {
    public:
        HeatmapModule(LayerList layers, ModuleConfig *config);

        void report_output(Buffer *buffer);

    protected:
        VisualizerWindow *window;

    MODULE_MEMBERS
};

#endif
