#ifndef image_input_module_h
#define image_input_module_h

#include "io/module.h"
#include "util/pointer.h"

class ImageInputModule : public Module {
    public:
        ImageInputModule(LayerList layers, ModuleConfig *config);
        virtual ~ImageInputModule() {
            this->gray.free();
            this->red.free();
            this->green.free();
            this->blue.free();
        }

        void feed_input(Buffer *buffer);

    private:
        bool transferred;
        int width;
        int height;
        Pointer<float> gray, red, green, blue;

    MODULE_MEMBERS
};

#endif
