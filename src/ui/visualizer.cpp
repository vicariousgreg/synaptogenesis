#include <climits>

#include "visualizer.h"
#include "visualizer_window.h"
#include "gui.h"
#include "model/layer.h"
#include "io/environment.h"

int Visualizer::instance_id = -1;

Visualizer *Visualizer::get_instance(bool init) {
    int id = Visualizer::instance_id;
    if ((id == -1 or id >= Frontend::instances.size())) {
        if (init) {
            new Visualizer();
            Visualizer::instance_id = Frontend::instances.size()-1;
        } else {
            Visualizer::instance_id = -1;
            return nullptr;
        }
    }
    return (Visualizer*)Frontend::instances[Visualizer::instance_id];
}

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

Visualizer::Visualizer() {
    this->visualizer_window = new VisualizerWindow();
    Frontend::set_window(this->visualizer_window);
}

Visualizer::~Visualizer() { }

bool Visualizer::add_input_layer(Layer *layer, std::string params) {
    LayerInfo* info;
    try {
        info = layer_map.at(layer);
    } catch (...) {
        info = new LayerInfo(layer);
        layer_map[layer] = info;
    }
    info->set_input();
    return true;
}

bool Visualizer::add_output_layer(Layer *layer, std::string params) {
    LayerInfo* info;
    try {
        info = layer_map.at(layer);
    } catch (...) {
        info = new LayerInfo(layer);
        layer_map[layer] = info;
    }
    info->set_output();
    return true;
}

void Visualizer::update(Environment *environment) {
    // Copy data over
    for (int i = 0; i < visualizer_window->layers.size(); ++i) {
        LayerInfo *info = visualizer_window->layers[i];
        if (info->get_output()) {
            guint8* data = visualizer_window->pixbufs[i]->get_pixels();
            Buffer *buffer = environment->buffer;
            Output *output = buffer->get_output(info->layer);
            OutputType output_type = environment->get_output_type(info->layer);

            for (int j = 0; j < info->layer->size; ++j)
                data[j*4 + 3] = 255 - convert(output[j], output_type);

            if (info->layer->rows == 1)
                for (int j = 1; j < 50; ++j)
                    for (int k = 0; k < info->layer->size; ++k)
                        data[(j*info->layer->size * 4) + (k*4 + 3)] = data[k*4 + 3];
        }
    }

    // Signal GUI to update
    this->gui->dispatcher.emit();
}
