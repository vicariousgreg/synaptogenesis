#ifndef dummy_output_module_h
#define dummy_output_module_h

#include "io/module/module.h"

class DummyOutputModule : public Module {
    public:
        DummyOutputModule(Layer *layer, ModuleConfig *config) : Module(layer) {}

        virtual IOTypeMask get_type() { return OUTPUT; }

    MODULE_MEMBERS
};

#endif
