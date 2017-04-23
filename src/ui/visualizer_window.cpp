#include "visualizer_window.h"

#include <iostream>

VisualizerWindow::VisualizerWindow() {
    grid = new Gtk::Grid();
    grid->set_row_spacing(1);
    grid->set_column_spacing(1);
    grid->override_background_color(Gdk::RGBA("DarkSlateGray"));
    this->add(*grid);
}


VisualizerWindow::~VisualizerWindow() {
    delete this->grid;
}

void VisualizerWindow::add_layer(LayerInfo* layer_info) {
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
        //Gtk::PositionType::POS_BOTTOM,
        Gtk::PositionType::POS_RIGHT,
        cols, rows);
    this->grid->show_all();
}

void VisualizerWindow::update() {
    for (int i = 0; i < images.size(); ++i)
        images[i]->set(this->pixbufs[i]);
}
