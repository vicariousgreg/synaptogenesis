#ifndef dummy_input_module_h
#define dummy_input_module_h

#include "io/module/module.h"

class DummyInputModule : public Module {
    public:
        DummyInputModule(Layer *layer, ModuleConfig *config) : Module(layer) {}


    MODULE_MEMBERS
};

#endif
