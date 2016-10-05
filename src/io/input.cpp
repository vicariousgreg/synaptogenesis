#include "io/input.h"
#include "io/random_input.h"
#include "io/image_input.h"

Input* build_input(Layer *layer, std::string type, std::string params) {
    if (type == "random")
        return new RandomInput(layer, params);
    else if (type == "image")
        return new ImageInput(layer, params);
    else
        throw "Unrecognized input type!";
}
