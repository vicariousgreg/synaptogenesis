#include "visualizer.h"
#include "gui.h"

#include <iostream>

Visualizer::Visualizer(Buffer *buffer) : buffer(buffer) {
    this->gui = new GUI();
}

Visualizer::~Visualizer() {
    delete this->gui;
}

void Visualizer::add_layer(Layer *layer, bool input, bool output) {
    this->gui->add_layer(LayerInfo(layer, input, output));
}

void Visualizer::launch() {
    this->gui->launch();
}

void Visualizer::update() {
    // Copy data over
    for (int i = 0; i < gui->layers.size(); ++i) {
        LayerInfo &info = gui->layers[i];
        if (info.output) {
            //guint8* data = gui->pixbufs[i]->get_pixels();
            guint8* data = gui->images[i]->get_pixbuf()->get_pixels();
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

    // Signal GUI to update
    this->gui->done = false;
    this->gui->signal_update.emit();
    while (!this->gui->done) ;
}
