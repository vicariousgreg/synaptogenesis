#ifndef dummy_output_module_h
#define dummy_output_module_h

#include "io/module.h"

class DummyOutputModule : public Module {
    public:
        DummyOutputModule(Layer *layer,
            std::string params, std::string &driver_type)
                : Module(layer, driver_type) {}

        virtual IOType get_type() { return OUTPUT; }
};

#endif
