#include "io/impl/image_input_module.h"
#include "util/tools.h"
#include "util/error_manager.h"

#define cimg_display 0
#include "CImg.h"

REGISTER_MODULE(ImageInputModule, "image_input");

ImageInputModule::ImageInputModule(LayerList layers, ModuleConfig *config)
        : Module(layers), transferred(false) {
    enforce_equal_layer_sizes("image_input");
    set_io_type(INPUT);

    std::string filename = config->get("filename", "");
    float scale = config->get_float("scale", 1);

    if (filename == "")
        ErrorManager::get_instance()->log_error(
            "Unspecified filename for image input module!");

    try {
        cimg_library::CImg<unsigned char> img(filename.c_str());
        width = img.width();
        height = img.height();

        // Check layer rows/columns and assign channels
        for (auto layer : layers) {
            if (width != layer->columns or height != layer->rows)
                ErrorManager::get_instance()->log_error(
                    "Image size does not match layer size!");
            auto channel =
                config->get_layer(layer)->get("channel", "gray");
            if (channel == "gray")       channel_map[layer] = GRAY;
            else if (channel == "red")   channel_map[layer] = RED;
            else if (channel == "green") channel_map[layer] = GREEN;
            else if (channel == "blue")  channel_map[layer] = BLUE;
            else
                ErrorManager::get_instance()->log_error(
                    "Unrecognized image channel: " + channel);
        }

        this->gray = Pointer<float>(width * height);
        this->red = Pointer<float>(width * height);
        this->green = Pointer<float>(width * height);
        this->blue = Pointer<float>(width * height);

        // Extract values from image
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                float red = scale * (float)img(c,r,0,0) / 255.0;
                float green = scale * (float)img(c,r,0,1) / 255.0;
                float blue = scale * (float)img(c,r,0,2) / 255.0;

                int index = (r*width) + c;
                this->gray[index] = (0.299*red + 0.587*green + 0.114*blue);
                this->red[index] = red;
                this->green[index] = green;
                this->blue[index] = blue;
            }
        }
    } catch (cimg_library::CImgIOException e) {
        ErrorManager::get_instance()->log_error(
            "Image " + filename + " could not be opened!\n");
    }
}

void ImageInputModule::feed_input(Buffer *buffer) {
    if (not this->transferred) {
        for (auto layer : layers) {
            switch (channel_map[layer]) {
                case GRAY:  buffer->set_input(layer, this->gray);
                case RED:   buffer->set_input(layer, this->red);
                case GREEN: buffer->set_input(layer, this->green);
                case BLUE:  buffer->set_input(layer, this->blue);
            }
            this->transferred = true;
        }
    }
}
