#include "io/input_module.h"
#include "io/random_input_module.h"
#include "io/image_input_module.h"

InputModule* build_input(Layer *layer, std::string type, std::string params) {
    if (type == "random")
        return new RandomInputModule(layer, params);
    else if (type == "image")
        return new ImageInputModule(layer, params);
    else
        throw "Unrecognized input type!";
}
