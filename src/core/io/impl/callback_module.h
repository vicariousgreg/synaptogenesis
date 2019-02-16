#ifndef callback_module_h
#define callback_module_h

#include "io/module.h"

class CallbackModule : public Module {
    public:
        CallbackModule(LayerList layers, ModuleConfig *config);

        void feed_input_impl(Buffer *buffer);
        void report_output_impl(Buffer *buffer);

    protected:
        std::map<Layer*, void (*)(int, int, void*)> callbacks;
        std::map<Layer*, int> ids;
        bool clear_first;

    MODULE_MEMBERS
};

#endif
