#include <climits>

#include "visualizer.h"
#include "visualizer_window.h"
#include "gui.h"
#include "network/layer.h"
#include "state/attributes.h"
#include "io/buffer.h"

std::string Visualizer::name = "visualizer";

Visualizer *Visualizer::get_instance(bool init) {
    auto instance = (Visualizer*)Frontend::get_instance(Visualizer::name);
    if (instance != nullptr)
        return instance;
    else if (init)
        return new Visualizer();
    else
        return nullptr;
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

bool Visualizer::add_input_layer(Layer *layer, std::string params) {
    LayerInfo* info;
    try {
        info = layer_map.at(layer);
        return false;
    } catch (std::out_of_range) {
        info = new LayerInfo(layer);
        this->add_layer(layer, info);
    }
    info->set_input();
    return true;
}

bool Visualizer::add_output_layer(Layer *layer, std::string params) {
    LayerInfo* info;
    try {
        info = layer_map.at(layer);
        return false;
    } catch (std::out_of_range) {
        info = new LayerInfo(layer);
        this->add_layer(layer, info);
    }
    info->set_output();
    return true;
}

void Visualizer::update(Buffer *buffer) {
    // Copy data over
    for (int i = 0; i < visualizer_window->layers.size(); ++i) {
        LayerInfo *info = visualizer_window->layers[i];
        if (info->get_output()) {
            guint8* data = visualizer_window->pixbufs[i]->get_pixels();
            Output *output = buffer->get_output(info->layer);
            OutputType output_type = Attributes::get_output_type(info->layer);

            for (int j = 0; j < info->layer->size; ++j) {
                float val = convert(output[j], output_type);
                data[j*4 + 0] = val;
                data[j*4 + 1] = val;
                data[j*4 + 2] = val;
            }

            if (info->layer->rows == 1)
                for (int j = 1; j < 50; ++j) {
                    int index = j*info->layer->size * 4;
                    for (int k = 0; k < info->layer->size; ++k) {
                        data[index + (k*4 + 0)] = data[k*4 + 0];
                        data[index + (k*4 + 1)] = data[k*4 + 1];
                        data[index + (k*4 + 2)] = data[k*4 + 2];
                    }
                }
        }
    }

    // Signal GUI to update
    this->gui->dispatcher.emit();
}
