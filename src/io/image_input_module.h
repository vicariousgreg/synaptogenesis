#ifndef image_input_module_h
#define image_input_module_h

#include "io/module.h"

class ImageInputModule : public Module {
    public:
        ImageInputModule(Layer *layer,
            std::string params, std::string &driver_type);
        void feed_input(Buffer *buffer);
        virtual IOType get_type() { return INPUT; }

    private:
        int width;
        int height;
        float* gray;
        float* red;
        float* green;
        float* blue;
};

#endif
