#include "input.h"
#include "random_input.h"

Input* build_input(Layer &layer, std::string type, std::string params) {
    if (type == "random")
        return new RandomInput(layer, params);
    else
        throw "Unrecognized input type!";
}
