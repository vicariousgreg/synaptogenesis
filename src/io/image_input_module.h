#ifndef image_input_module_h
#define image_input_module_h

#include "io/input_module.h"

class ImageInputModule : public InputModule {
    public:
        ImageInputModule(Layer *layer, std::string params);
        void feed_input(Buffer *buffer);

        int width;
        int height;
        float* gray;
        float* red;
        float* green;
        float* blue;
};

#endif
