#ifndef random_input_h
#define random_input_h

#include "io/input.h"

class RandomInput : public Input {
    public:
        RandomInput(Layer *layer, std::string params);
        void feed_input(Buffer *buffer);

        float max_value;
        float* buffer;
};

#endif
