#ifndef visualizer_output_module_h
#define visualizer_output_module_h

#include "io/module/module.h"
#include "visualizer.h"

class VisualizerOutputModule : public Module {
    public:
        VisualizerOutputModule(Layer *layer, std::string params)
            : Module(layer) {
            Visualizer::get_instance(true)->add_output_layer(layer);
        }

        virtual IOTypeMask get_type() { return OUTPUT; }
};

#endif
