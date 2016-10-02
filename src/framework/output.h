#ifndef output_h
#define output_h

#include <string>

#include "state.h"
#include "model.h"

class Output {
    public:
        Output(Layer *layer) : layer(layer) { }

        virtual void report_output(State *state) = 0;

        Layer *layer;
};

Output* build_output(Layer *layer, std::string type, std::string params);

#endif
