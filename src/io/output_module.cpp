#include "io/output_module.h"
#include "io/print_output_module.h"

OutputModule* build_output(Layer *layer, std::string type,
        std::string params, std::string &driver_type) {
    if (type == "print")
        return new PrintOutputModule(layer, params, driver_type);
    else
        throw "Unrecognized output type!";
}
