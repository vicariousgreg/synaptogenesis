#include "gui.h"

#include <iostream>

GUI::GUI() {
    // Mock arguments
    int argc = 1;
    this->argv = (char**)malloc(sizeof(char*));
    this->argv[0] = " ";
    app =
        Gtk::Application::create(argc, this->argv,
                "org.gtkmm.examples.base");

    window = new Gtk::Window();
    grid = new Gtk::Grid();
    window->add(*grid);
    dispatcher.connect(sigc::mem_fun(*this, &GUI::update));
}


GUI::~GUI() {
    delete this->window;
    delete this->grid;
    free(this->argv);
}

void GUI::add_layer(LayerInfo* layer_info) {
    this->layers.push_back(layer_info);
    int cols = layer_info->layer->columns;
    int rows = layer_info->layer->rows;

    // If there is only one row, expand it
    rows = (rows == 1) ? 50 : rows;

    auto pix = Gdk::Pixbuf::create(
            Gdk::Colorspace::COLORSPACE_RGB,
            true,
            8,
            cols,
            rows);
    guint8* data = pix->get_pixels();
    for (int j = 0; j < rows*cols; ++j) {
        data[j*4 + 0] = 0;
        data[j*4 + 1] = 0;
        data[j*4 + 2] = 0;
        data[j*4 + 3] = 255;
    }
    auto image = new Gtk::Image(pix);

    this->pixbufs.push_back(pix);
    this->images.push_back(image);

    //this->grid->add(*Gtk::manage(image));
    this->grid->attach_next_to(
        *Gtk::manage(image),
        Gtk::PositionType::POS_BOTTOM,
        //Gtk::PositionType::POS_RIGHT,
        cols, rows);
    this->grid->show_all();
}

void GUI::launch() {
    for (int i = 0; i < images.size(); ++i)
        images[i]->show();
    app->run(*window);
}

void GUI::update() {
    for (int i = 0; i < images.size(); ++i)
        images[i]->set(this->pixbufs[i]);
}
