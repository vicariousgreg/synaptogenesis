#ifndef image_input_h
#define image_input_h

#include "input.h"

class ImageInput : public Input {
    public:
        ImageInput(Layer &layer, std::string params);
        void feed_input(State *state);

        int width;
        int height;
        float* gray;
        float* red;
        float* green;
        float* blue;
};

#endif
