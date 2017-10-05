#ifndef callback_module_h
#define callback_module_h

#include "io/module.h"
#include "util/error_manager.h"
#include "visualizer_window.h"

class CallbackModule : public Module {
    public:
        CallbackModule(LayerList layers, ModuleConfig *config);

        void feed_input(Buffer *buffer);
        void feed_expected(Buffer *buffer);
        void report_output(Buffer *buffer);

    protected:
        std::map<Layer*, void (*)(int, int, void*)> callbacks;
        std::map<Layer*, int> ids;

    MODULE_MEMBERS
};

#endif
