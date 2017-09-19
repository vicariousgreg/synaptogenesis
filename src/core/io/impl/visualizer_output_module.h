#ifndef visualizer_output_module_h
#define visualizer_output_module_h

#include "io/module.h"
#include "util/error_manager.h"
#include "visualizer.h"

class VisualizerOutputModule : public Module {
    public:
        VisualizerOutputModule(LayerList layers, ModuleConfig *config)
            : Module(layers) {
            for (auto layer : layers)
                if (not Visualizer::get_instance(true)
                        ->add_output_layer(layer,
                            config->get_property("params", "")))
                    ErrorManager::get_instance()->log_error(
                        "Failed to add layer to Visualizer!");
        }


    MODULE_MEMBERS
};

#endif
