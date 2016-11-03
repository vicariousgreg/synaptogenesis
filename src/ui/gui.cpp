#include "gui.h"

#include <iostream>

GUI::GUI(Buffer *buffer) : buffer(buffer) {
    // Mock arguments
    int argc = 1;
    char **argv = (char**)malloc(sizeof(char*));
    argv[0] = " ";
    app =
        Gtk::Application::create(argc, argv,
                "org.gtkmm.examples.base");

    window = new Gtk::Window();
    grid = new Gtk::Grid();
    window->add(*grid);
    dispatcher.connect(sigc::mem_fun(*this, &GUI::update));

    free(argv);
}


GUI::~GUI() {
    delete this->window;
    delete this->grid;
}

void GUI::add_layer(LayerInfo layer_info) {
    this->layers.push_back(layer_info);
    auto pix = Gdk::Pixbuf::create(
            Gdk::Colorspace::COLORSPACE_RGB,
            true,
            8,
            layer_info.layer->columns,
            layer_info.layer->rows);
    guint8* data = pix->get_pixels();
    for (int j = 0; j < layer_info.layer->size; ++j) {
        data[j*4 + 0] = 0;
        data[j*4 + 1] = 0;
        data[j*4 + 2] = 0;
        data[j*4 + 3] = 255;
    }
    auto image = new Gtk::Image(pix);

    this->pixbufs.push_back(pix);
    this->images.push_back(image);

    this->grid->add(*Gtk::manage(image));
    this->grid->show_all();

}

void GUI::launch() {
    for (int i = 0; i < images.size(); ++i)
        images[i]->show();
    app->run(*window);
}

void GUI::update() {
    // Copy data over
    for (int i = 0; i < layers.size(); ++i) {
        LayerInfo &info = layers[i];
        if (info.output) {
            guint8* data = pixbufs[i]->get_pixels();
            //guint8* data = images[i]->get_pixbuf()->get_pixels();
            int output_index = info.layer->output_index;
            Output *output = buffer->get_output() + output_index;
            for (int j = 0; j < info.layer->size; ++j) {
                guint8 val = (guint8)output[j].i;
                data[j*4 + 0] = val;
                data[j*4 + 1] = val;
                data[j*4 + 2] = val;
                data[j*4 + 3] = 255;
            }
        }
    }

    for (int i = 0; i < images.size(); ++i) {
        images[i]->set(this->pixbufs[i]);
    }
}
