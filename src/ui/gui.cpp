#include "gui.h"

#include <iostream>

GUI::GUI() {
    int argc = 0;
    char **argv = NULL;
    app =
        Gtk::Application::create(argc, argv,
                "org.gtkmm.examples.base");

    window = new Gtk::Window();
    window->set_default_size(200, 200);

    signal_update.connect(sigc::mem_fun(
        this, &GUI::update) );
}


GUI::~GUI() {
    delete this->window;
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
    /*
    for (int j = 0; j < layer_info.layer->size; ++j) {
        data[j*4 + 0] = 0;
        data[j*4 + 1] = 0;
        data[j*4 + 2] = 0;
        data[j*4 + 3] = 255;
    }
    */
    auto image = new Gtk::Image(pix);

    this->pixbufs.push_back(pix);
    this->images.push_back(image);

    this->window->add(*Gtk::manage(image));

}

void GUI::launch() {
    for (int i = 0; i < images.size(); ++i)
        images[i]->show();
    app->run(*window);
}

void GUI::update() {
    for (int i = 0; i < images.size(); ++i) {
        images[i]->set(this->pixbufs[i]);
    }

    this->done = true;
}
