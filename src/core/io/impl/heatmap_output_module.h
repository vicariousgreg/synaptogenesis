#ifndef heatmap_output_module_h
#define heatmap_output_module_h

#include "io/module.h"
#include "util/error_manager.h"
#include "heatmap.h"

class HeatmapOutputModule : public Module {
    public:
        HeatmapOutputModule(LayerList layers, ModuleConfig *config)
            : Module(layers) {
            for (auto layer : layers)
                if (not Heatmap::get_instance(true)
                        ->add_output_layer(layer,
                            config->get_property("params", "")))
                    ErrorManager::get_instance()->log_error(
                        "Failed to add layer to Heatmap!");
        }


    MODULE_MEMBERS
};

#endif
