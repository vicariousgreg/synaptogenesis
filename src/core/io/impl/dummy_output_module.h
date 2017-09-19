#ifndef dummy_output_module_h
#define dummy_output_module_h

#include "io/module.h"

class DummyOutputModule : public Module {
    public:
        DummyOutputModule(LayerList layers, ModuleConfig *config) : Module(layers) {}


    MODULE_MEMBERS
};

#endif
