#ifndef visualizer_input_module_h
#define visualizer_input_module_h

#include "io/module/module.h"
#include "visualizer.h"

class VisualizerInputModule : public Module {
    public:
        VisualizerInputModule(Layer *layer, std::string params)
            : Module(layer) {
            Visualizer::get_instance(true)->add_input_layer(layer);
        }

        virtual IOTypeMask get_type() { return INPUT; }
};

#endif
