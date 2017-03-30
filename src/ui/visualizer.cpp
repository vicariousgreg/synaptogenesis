#include <climits>

#include "io/environment.h"
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

Visualizer::Visualizer(Environment *environment) : environment(environment) {
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
            guint8* data = gui->pixbufs[i]->get_pixels();
            Buffer *buffer = environment->buffer;
            Output *output = buffer->get_output(info.layer);
            OutputType output_type = environment->get_output_type(info.layer);

            for (int j = 0; j < info.layer->size; ++j)
                data[j*4 + 3] = 255 - convert(output[j], output_type);

            if (info.layer->rows == 1)
                for (int j = 1; j < 50; ++j)
                    for (int k = 0; k < info.layer->size; ++k)
                        data[(j*info.layer->size * 4) + (k*4 + 3)] = data[k*4 + 3];
        }
    }

    // Signal GUI to update
    this->gui->dispatcher.emit();
}
