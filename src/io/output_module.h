#ifndef output_module_h
#define output_module_h

#include <string>

#include "model/layer.h"
#include "io/buffer.h"

class OutputModule {
    public:
        OutputModule(Layer *layer) : layer(layer) { }

        virtual void report_output(Buffer *buffer) = 0;

        Layer *layer;
};

OutputModule* build_output(Layer *layer, std::string type, std::string params);

#endif
