#include <cstdlib>

#include "io/random_input.h"
#include "tools.h"

RandomInput::RandomInput(Layer *layer, std::string params) : Input(layer) {
    this->max_value = strtof(params.c_str(), NULL);
    if (this->max_value == 0.0)
        throw "Invalid max value for random input generator!";
}

void RandomInput::feed_input(Buffer *buffer) {
    int offset = this->layer->index;
    float *input = buffer->get_input();
    for (int nid = 0 ; nid < this->layer->size; ++nid) {
        input[offset + nid] = fRand(0, this->max_value);
    }
}
