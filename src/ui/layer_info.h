#ifndef layer_info_h
#define layer_info_h

#include "model/layer.h"

class Structure;

class LayerInfo {
    public:
        LayerInfo(Layer* layer, bool input, bool output)
            : layer(layer), structure(layer->structure),
              input(input), output(output) {}

        Layer *layer;
        Structure *structure;
        bool input, output;
};

#endif
