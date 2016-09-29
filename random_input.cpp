#include <cstdlib>

#include "random_input.h"
#include "tools.h"

RandomInput::RandomInput(Layer &layer, std::string params) : Input(layer) {
    this->max_value = strtof(params.c_str(), NULL);
    if (this->max_value == 0.0)
        throw "Invalud max value for random input generator!";
    this->buffer = (float*)malloc(this->layer.size * sizeof(float));
}

void RandomInput::feed_input(State *state) {
    for (int nid = 0 ; nid < this->layer.size; ++nid) {
        this->buffer[nid] = fRand(0, this->max_value);
    }
    state->set_input(this->layer.id, this->buffer);
}
