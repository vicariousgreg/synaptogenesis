#ifndef output_h
#define output_h

#include <string>

#include "model/model.h"
#include "io/buffer.h"

class Output {
    public:
        Output(Layer *layer) : layer(layer) { }

        virtual void report_output(Buffer *buffer) = 0;

        Layer *layer;
};

Output* build_output(Layer *layer, std::string type, std::string params);

#endif
