#include <cstdlib>

#include "io/random_input_module.h"
#include "tools.h"

RandomInputModule::RandomInputModule(Layer *layer, std::string params) : InputModule(layer) {
    this->max_value = strtof(params.c_str(), NULL);
    if (this->max_value == 0.0)
        throw "Invalid max value for random input generator!";
}

void RandomInputModule::feed_input(Buffer *buffer) {
    int offset = this->layer->index;
    float *input = buffer->get_input();
    for (int nid = 0 ; nid < this->layer->size; ++nid) {
        input[offset + nid] = fRand(0, this->max_value);
    }
}
