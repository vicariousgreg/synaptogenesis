#ifndef dummy_module_h
#define dummy_module_h

#include "io/module.h"

class DummyModule : public Module {
    public:
        DummyModule(LayerList layers, ModuleConfig *config)
                : Module(layers, config) {
            for (auto layer : layers) {
                auto layer_config = config->get_layer(layer);

                if (layer_config->get_bool("input", false))
                    set_io_type(layer, get_io_type(layer) | INPUT);

                if (layer_config->get_bool("expected", false))
                    set_io_type(layer, get_io_type(layer) | EXPECTED);

                if (layer_config->get_bool("output", false))
                    set_io_type(layer, get_io_type(layer) | OUTPUT);

                // Use output as default
                if (get_io_type(layer) == 0)
                    set_io_type(layer, OUTPUT);
            }
        }


    MODULE_MEMBERS
};

#endif
