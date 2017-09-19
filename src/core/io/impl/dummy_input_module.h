#ifndef dummy_input_module_h
#define dummy_input_module_h

#include "io/module.h"

class DummyInputModule : public Module {
    public:
        DummyInputModule(LayerList layers, ModuleConfig *config)
                : Module(layers) {
            for (auto layer : layers)
                set_io_type(INPUT);
        }


    MODULE_MEMBERS
};

#endif
