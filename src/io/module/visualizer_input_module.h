#ifndef visualizer_input_module_h
#define visualizer_input_module_h

#include "io/module/module.h"

class VisualizerInputModule : public Module {
    public:
        VisualizerInputModule(Layer *layer, std::string params) : Module(layer) {}

        virtual IOType get_type() { return INPUT; }
};

#endif
