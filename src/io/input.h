#ifndef input_h
#define input_h

#include <string>

#include "io/buffer.h"
#include "model/model.h"

class Input {
    public:
        Input(Layer *layer) : layer(layer) { }

        virtual void feed_input(Buffer *buffer) = 0;

        Layer *layer;
};

Input* build_input(Layer *layer, std::string type, std::string params);

#endif
