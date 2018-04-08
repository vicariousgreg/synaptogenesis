#ifdef __GUI__

#ifndef saccade_module_h
#define saccade_module_h

#include "io/module.h"

class SaccadeWindow;

class SaccadeModule : public Module {
    public:
        SaccadeModule(LayerList layers, ModuleConfig *config);

        void feed_input_impl(Buffer *buffer);
        void report_output_impl(Buffer *buffer);

        void init();
        void update(Buffer *buffer);

    private:
        std::map<Layer*, bool> central;
        SaccadeWindow *window;

    MODULE_MEMBERS
};

#endif

#endif
