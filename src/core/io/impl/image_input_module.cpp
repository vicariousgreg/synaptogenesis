#include <cstdlib>
#include <iostream>

#include "io/impl/image_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#define cimg_display 0
#include "CImg.h"

REGISTER_MODULE(ImageInputModule, "image_input", INPUT);

ImageInputModule::ImageInputModule(Layer *layer, ModuleConfig *config)
        : Module(layer), transferred(false) {
    std::string filename = config->get_property("filename");
    try {
        cimg_library::CImg<unsigned char> img(filename.c_str());
        width = img.width();
        height = img.height();

        if (width != layer->columns or height != layer->rows) {
            ErrorManager::get_instance()->log_error(
                "Image size does not match layer size!");
        }

        this->gray = Pointer<float>(width * height);
        this->red = Pointer<float>(width * height);
        this->green = Pointer<float>(width * height);
        this->blue = Pointer<float>(width * height);

        float factor = 10;

        // Extract values from image
        //std::cout << width << "x" << height << std::endl;
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                float red = factor * (float)img(c,r,0,0) / 255.0;
                float green = factor * (float)img(c,r,0,1) / 255.0;
                float blue = factor * (float)img(c,r,0,2) / 255.0;

                int index = (r*width) + c;
                this->gray[index] = (0.299*red + 0.587*green + 0.114*blue);
                this->red[index] = red;
                this->green[index] = green;
                this->blue[index] = blue;
                /*
                std::cout << "(" << r << "," << c << ") ="
                    << " R" << red
                    << " G" << green
                    << " B" << blue
                    << " : " << gray << std::endl;
                */
            }
        }
    } catch (cimg_library::CImgIOException e) {
        printf("Image %s could not be opened!\n", filename.c_str());
        ErrorManager::get_instance()->log_error(
            "Could not construct image input driver!");
    }
}

void ImageInputModule::feed_input(Buffer *buffer) {
    if (not this->transferred) {
        buffer->set_input(this->layer, this->gray);
        this->transferred = true;
    }
}
