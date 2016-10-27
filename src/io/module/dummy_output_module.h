#ifndef dummy_output_module_h
#define dummy_output_module_h

#include "io/module/module.h"

class DummyOutputModule : public Module {
    public:
        DummyOutputModule(Layer *layer, std::string params) : Module(layer) {}

        virtual IOType get_type() { return OUTPUT; }
};

#endif
