#ifndef input_h
#define input_h

#include <string>

#include "state.h"
#include "model.h"

class Input {
    public:
        Input(Layer &layer) : layer(layer) { }

        virtual void feed_input(State *state) = 0;

        Layer &layer;
};

Input* build_input(Layer &layer, std::string type, std::string params);

#endif
