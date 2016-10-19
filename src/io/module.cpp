#include "io/module.h"
#include "io/print_output_module.h"
#include "io/random_input_module.h"
#include "io/image_input_module.h"

Module* build_module(Layer *layer, std::string type,
        std::string params, std::string &driver_type) {
    if (type == "random_input")
        return new RandomInputModule(layer, params, driver_type);
    else if (type == "image_input")
        return new ImageInputModule(layer, params, driver_type);
    else if (type == "print_output")
        return new PrintOutputModule(layer, params, driver_type);
    else
        throw "Unrecognized output type!";
}
