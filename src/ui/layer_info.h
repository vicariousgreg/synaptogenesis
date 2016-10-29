#ifndef layer_info_h
#define layer_info_h

#include "model/layer.h"

class LayerInfo {
    public:
        LayerInfo(Layer* layer, bool input, bool output)
            : layer(layer), input(input), output(output) {}

        Layer *layer;
        bool input, output;
};

#endif
