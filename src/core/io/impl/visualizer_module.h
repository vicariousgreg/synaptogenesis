#ifndef visualizer_module_h
#define visualizer_module_h

#include "io/module.h"
#include "util/error_manager.h"
#include "visualizer_window.h"

class VisualizerModule : public Module {
    public:
        VisualizerModule(LayerList layers, ModuleConfig *config);

        void feed_input(Buffer *buffer);
        void report_output(Buffer *buffer);

    protected:
        VisualizerWindow *window;

    MODULE_MEMBERS
};

#endif
