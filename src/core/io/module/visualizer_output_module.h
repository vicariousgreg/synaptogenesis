#ifndef visualizer_output_module_h
#define visualizer_output_module_h

#include "io/module/module.h"
#include "util/error_manager.h"
#include "visualizer.h"

class VisualizerOutputModule : public Module {
    public:
        VisualizerOutputModule(Layer *layer, ModuleConfig *config)
            : Module(layer) {
            if (not Visualizer::get_instance(true)
                    ->add_output_layer(layer, config->get_property("params")))
                ErrorManager::get_instance()->log_error(
                    "Failed to add layer to Visualizer!");
        }

        virtual IOTypeMask get_type() { return OUTPUT; }

    MODULE_MEMBERS
};

#endif
