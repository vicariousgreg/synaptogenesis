#ifndef visualizer_input_module_h
#define visualizer_input_module_h

#include "io/module/module.h"
#include "util/error_manager.h"
#include "visualizer.h"

class VisualizerInputModule : public Module {
    public:
        VisualizerInputModule(Layer *layer, ModuleConfig *config)
            : Module(layer) {
            if (not Visualizer::get_instance(true)
                    ->add_input_layer(layer, config->get_property("params")))
                ErrorManager::get_instance()->log_error(
                    "Failed to add layer to Visualizer!");
        }


    MODULE_MEMBERS
};

#endif
