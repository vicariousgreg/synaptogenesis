#ifndef visualizer_input_module_h
#define visualizer_input_module_h

#include "io/module/module.h"
#include "util/error_manager.h"
#include "visualizer.h"

class VisualizerInputModule : public Module {
    public:
        VisualizerInputModule(Layer *layer, std::string params)
            : Module(layer) {
            if (not Visualizer::get_instance(true)
                    ->add_input_layer(layer, params))
                ErrorManager::get_instance()->log_error(
                    "Failed to add layer to Visualizer!");
        }

        virtual IOTypeMask get_type() { return INPUT; }
};

#endif
