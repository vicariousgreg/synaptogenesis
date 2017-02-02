#include <climits>

#include "visualizer.h"
#include "gui.h"

static guint8 convert(Output out, OutputType type) {
    switch (type) {
        case FLOAT:
            return 255 * out.f;
        case INT:
            return 255 * float(out.i) / INT_MAX;
        case BIT:
            return (out.i > INT_MAX) ? 255 : (out.i >> 23);
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
    OutputType output_type = buffer->output_type;

    // Copy data over
    for (int i = 0; i < gui->layers.size(); ++i) {
        LayerInfo &info = gui->layers[i];
        if (info.output) {
            guint8* data = gui->pixbufs[i]->get_pixels();
            Output *output = buffer->get_output() + info.layer->get_output_index();

            for (int j = 0; j < info.layer->size; ++j)
                data[j*4 + 3] = 255 - convert(output[j], output_type);
        }
    }

    // Signal GUI to update
    this->gui->dispatcher.emit();
}
