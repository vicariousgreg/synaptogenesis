#include "visualizer.h"
#include "gui.h"

#include <iostream>

static guint8 convert(Output out, OutputType type) {
    switch (type) {
        case FLOAT:
            return 255 * out.f;
            break;
        case INT:
            return 255 * ((out.i & 0xFF) / 0xFF);
            break;
        case BIT:
            unsigned int val = out.i;
            val = ((val & 0x55555555) << 1) | ((val & 0xAAAAAAAA) >> 1);
            val = ((val & 0x33333333) << 2) | ((val & 0xCCCCCCCC) >> 2);
            val = ((val & 0x0F0F0F0F) << 4) | ((val & 0xF0F0F0F0) >> 4);
            val = ((val & 0x00FF00FF) << 8) | ((val & 0xFF00FF00) >> 8);
            val = ((val & 0x0000FFFF) << 16) | ((val & 0xFFFF0000) >> 16);
            return val >> ((sizeof(int) - 1) * 8);
            break;
    }
}

Visualizer::Visualizer(Buffer *buffer) : buffer(buffer) {
    this->gui = new GUI(buffer);
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
            guint8* data = gui->pixbufs[i]->get_pixels();
            int output_index = info.layer->output_index;
            Output *output = buffer->get_output() + output_index;
            for (int j = 0; j < info.layer->size; ++j) {
                guint8 val = convert(output[j], buffer->output_type);
                data[j*4 + 0] = val;
                data[j*4 + 1] = val;
                data[j*4 + 2] = val;
                data[j*4 + 3] = 255;
            }
        }
    }

    // Signal GUI to update
    this->gui->dispatcher.emit();
}
