#ifndef heatmap_output_module_h
#define heatmap_output_module_h

#include "io/module/module.h"
#include "util/error_manager.h"
#include "heatmap.h"

class HeatmapOutputModule : public Module {
    public:
        HeatmapOutputModule(Layer *layer, std::string params)
            : Module(layer) {
            if (not Heatmap::get_instance(true)
                    ->add_output_layer(layer, params))
                ErrorManager::get_instance()->log_error(
                    "Failed to add layer to Heatmap!");
        }

        virtual IOTypeMask get_type() { return OUTPUT; }

    MODULE_MEMBERS
};

#endif
