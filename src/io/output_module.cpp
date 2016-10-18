#include "io/output_module.h"
#include "io/spike_print_output_module.h"
#include "io/float_print_output_module.h"

OutputModule* build_output(Layer *layer, std::string type, std::string params) {
    if (type == "print_spike")
        return new SpikePrintOutputModule(layer, params);
    else if (type == "print_float")
        return new FloatPrintOutputModule(layer, params);
    else
        throw "Unrecognized output type!";
}
