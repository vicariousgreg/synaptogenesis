#ifndef dummy_input_module_h
#define dummy_input_module_h

#include "io/module.h"

class DummyInputModule : public Module {
    public:
        DummyInputModule(Layer *layer, std::string params) : Module(layer) {}

        virtual IOType get_type() { return INPUT; }
};

#endif
