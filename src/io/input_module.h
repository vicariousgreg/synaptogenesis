#ifndef input_module_h
#define input_module_h

#include <string>

#include "io/buffer.h"
#include "model/layer.h"

class InputModule {
    public:
        InputModule(Layer *layer) : layer(layer) { }

        virtual void feed_input(Buffer *buffer) = 0;

        Layer *layer;
};

InputModule* build_input(Layer *layer, std::string type, std::string params);

#endif
