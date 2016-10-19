#include <cstdlib>
#include <iostream>

#include "io/image_input_module.h"
#include "tools.h"

#define cimg_display 0
#include "libs/CImg.h"

ImageInputModule::ImageInputModule(Layer *layer,
        std::string params, std::string &driver_type)
        : Module(layer, driver_type) {
    try {
        cimg_library::CImg<unsigned char> img(params.c_str());
        width = img.width();
        height = img.height();

        if (width != layer->columns or height != layer->rows) {
            throw "Image size does not match layer size!";
        }

        this->gray = (float*)malloc(width * height * sizeof(float));
        this->red = (float*)malloc(width * height * sizeof(float));
        this->green = (float*)malloc(width * height * sizeof(float));
        this->blue = (float*)malloc(width * height * sizeof(float));

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
        printf("Image %s could not be opened!\n", params.c_str());
        throw "Could not construct image input driver!";
    }
}

void ImageInputModule::feed_input(Buffer *buffer) {
    buffer->set_input(this->layer->input_index, this->layer->size, this->gray);
}
