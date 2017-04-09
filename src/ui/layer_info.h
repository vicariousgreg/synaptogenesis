#ifndef layer_info_h
#define layer_info_h

#include "model/layer.h"

class Structure;

class LayerInfo {
    public:
        LayerInfo(Layer* layer)
            : layer(layer), structure(layer->structure),
              input(false), output(false) {}

        void set_input() { input = true; }
        void set_output() { output = true; }
        bool get_input() { return input; }
        bool get_output() { return output; }

        Layer* const layer;
        Structure* const structure;

    private:
        bool input, output;
};

#endif
