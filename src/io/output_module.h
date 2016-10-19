#ifndef output_module_h
#define output_module_h

#include <string>

#include "model/layer.h"
#include "io/buffer.h"

class OutputModule {
    public:
        OutputModule(Layer *layer, std::string &driver_type)
                : layer(layer), driver_type(driver_type) { }

        virtual void report_output(Buffer *buffer) = 0;

        Layer *layer;
        std::string &driver_type;
};

OutputModule* build_output(Layer *layer, std::string type,
    std::string params, std::string &driver_type);

#endif
