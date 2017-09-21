#ifndef dummy_module_h
#define dummy_module_h

#include "io/module.h"

class DummyModule : public Module {
    public:
        DummyModule(LayerList layers, ModuleConfig *config)
                : Module(layers) {
            for (auto layer : layers) {
                auto params =
                    config->get_layer(layer)->get_property("params", "output");

                if (params == "input")
                    set_io_type(layer, INPUT);
                else if (params == "output")
                    set_io_type(layer, OUTPUT);
                else
                    ErrorManager::get_instance()->log_error(
                        "Unrecognized layer type: " + params
                        + " in DummyModule!");
            }
        }


    MODULE_MEMBERS
};

#endif
