#ifndef dummy_module_h
#define dummy_module_h

#include "io/module.h"

class DummyModule : public Module {
    public:
        DummyModule(LayerList layers, ModuleConfig *config)
                : Module(layers, config) {
            // Use output as default
            set_default_io_type(OUTPUT);
        }


    MODULE_MEMBERS
};

#endif
