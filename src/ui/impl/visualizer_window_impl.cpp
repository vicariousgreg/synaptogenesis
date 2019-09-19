#include <math.h>

#include "visualizer_window.h"
#include "visualizer_window_impl.h"

#include "network/structure.h"

static float convert(Output out, OutputType type) {
    switch (type) {
        case FLOAT:
            return 255 * MIN(1.0, out.f);
        case INT:
            return 255 * float(out.i) / INT_MAX;
        case BIT:
            return (out.i > INT_MAX) ? 255 : (out.i >> 23);
    }
}

VisualizerWindowImpl::VisualizerWindowImpl(PropertyConfig *config) :
          colored(config->get_bool("colored", false)),
          negative(config->get_bool("negative", false)),
          decay(config->get_bool("decay", false)),
          bump(config->get_int("bump", 16)),
          color_window(config->get_int("window", 256)),
          freq_r(3.8 / color_window),
          freq_g(freq_r * 2),
          freq_b(freq_r * 3) {
    grid = new Gtk::Grid();
    grid->set_row_spacing(1);
    grid->set_column_spacing(1);
    grid->override_background_color(Gdk::RGBA("DarkSlateGray"));
    this->add(*grid);
}

VisualizerWindowImpl::~VisualizerWindowImpl() {
    delete this->grid;
    for (auto pair : hues) free(pair.second);
}

void VisualizerWindowImpl::add_layer(Layer *layer, IOTypeMask io_type) {
    this->layer_indices[layer] = this->layers.size();
    this->layers.push_back(layer);
    int cols = layer->columns;
    int rows = layer->rows;

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

    if (colored)  {
        int *layer_hues = (int*) calloc(layer->size, sizeof(int));
        this->hues[layer->id] = layer_hues;
        for (int j = 0; j < layer->size; ++j)
            layer_hues[j] = color_window;
    }

}

void VisualizerWindowImpl::update() {
    for (int i = 0; i < images.size(); ++i)
        images[i]->set(this->pixbufs[i]);
}

void VisualizerWindowImpl::feed_input(Layer *layer, float *input) {
}

void VisualizerWindowImpl::report_output(Layer *layer,
        Output *output, OutputType output_type) {
    guint8* data =
        this->pixbufs[layer_indices[layer]]->get_pixels();
    int *layer_hues = hues[layer->id];

    for (int j = 0; j < layer->size; ++j) {
        if (colored and output_type == BIT) {
            int hue = layer_hues[j] = (output[j].i & 1)
                ? (decay ? std::max(0, layer_hues[j] - bump) : 0)
                : std::min(layer_hues[j] + 1, color_window+1);

            if (hue == color_window+1) {
                continue;
            } else if (hue == color_window) {
                data[j*4 + 0] = 0;
                data[j*4 + 1] = 0;
                data[j*4 + 2] = 0;
            } else {
                data[j*4 + 0] = sin(float(hue) * freq_r + 1) * 127 + 128;
                data[j*4 + 1] = sin(float(hue) * freq_g + 3) * 127 + 128;
                data[j*4 + 2] = sin(float(hue) * freq_b + 5) * 127 + 128;
            }
        } else if (negative) {
            float val = convert(output[j], output_type);
            if (val >= 0.f) {
                data[j*4 + 0] = 0;
                data[j*4 + 1] = val;
                data[j*4 + 2] = 0;
            } else {
                data[j*4 + 0] = -val;
                data[j*4 + 1] = 0;
                data[j*4 + 2] = 0;
            }
        } else {
            guint8 val = convert(output[j], output_type);
            data[j*4 + 0] = val;
            data[j*4 + 1] = val;
            data[j*4 + 2] = val;
        }
    }

    if (layer->rows == 1) {
        for (int j = 1; j < 50; ++j) {
            int index = j*layer->size * 4;
            for (int k = 0; k < layer->size; ++k) {
                data[index + (k*4 + 0)] = data[k*4 + 0];
                data[index + (k*4 + 1)] = data[k*4 + 1];
                data[index + (k*4 + 2)] = data[k*4 + 2];
            }
        }
    }
}

HeatmapWindowImpl::HeatmapWindowImpl(PropertyConfig *config)
        : VisualizerWindowImpl(config),
          iterations(0),
          integration_window(config->get_int("window", 1000)),
          linear(config->get_bool("linear", false)),
          stats(config->get_bool("stats", true)) {
    if (integration_window < 1)
        LOG_ERROR("Invalid integrationwindow in HeatmapModule: "
            + std::to_string(integration_window));

    if (stats) {
        label = new Gtk::Label();
        label->override_color(Gdk::RGBA("White"));
        label->override_font(Pango::FontDescription("monospace"));
        this->grid->attach_next_to(
            *Gtk::manage(label),
            //Gtk::PositionType::POS_BOTTOM,
            Gtk::PositionType::POS_RIGHT,
            1, 1);
    }
}

HeatmapWindowImpl::~HeatmapWindowImpl() {
    for (auto pair : output_count_map) free(pair.second);
}

void HeatmapWindowImpl::update() {
    VisualizerWindowImpl::update();
    if (not stats) return;

    if (iterations % integration_window == (integration_window-1)) {
        std::string text;
        for (auto layer : layers) {
            float* output_count = output_count_map[layer->id];
            float max = output_count[0];
            float min = output_count[0];
            float sum = 0.0;
            int count = 0;

            for (int j = 0; j < layer->size; ++j) {
                float c = output_count[j];
                max = MAX(c, max);
                sum += c;
                if (c > 0) {
                    ++count;
                    min = MAX(1, MIN(c, min));
                }
            }
            float avg = sum / layer->size;
            float avg_non_silent = sum / count;
            max = max * 1000.0 / integration_window;
            min = min * 1000.0 / integration_window;
            avg = avg * 1000.0 / integration_window;
            avg_non_silent = avg_non_silent * 1000.0 / integration_window;

            float percentage = 100.0 * count / layer->size;

            text += layer->str() + "\n";
            text += "Spiked: " + std::to_string(count) +
                " / " + std::to_string(layer->size) +
                "(" + std::to_string(percentage) + "%)\n";
            text += "Max: " + std::to_string(int(max)) + "\n";
            text += "Min: " + std::to_string(int(min)) + "\n";
            text += "Avg: " + std::to_string(avg) + "\n";
            text += "     " + std::to_string(avg_non_silent) + "\n" + "\n";
        }
        label->set_text(text);
    }
}

void HeatmapWindowImpl::add_layer(Layer *layer, IOTypeMask io_type) {
    VisualizerWindowImpl::add_layer(layer, io_type);
    this->output_count_map[layer->id] = (float*) calloc(layer->size, sizeof(float));
}

void HeatmapWindowImpl::feed_input(Layer *layer, float *input) {
}

void HeatmapWindowImpl::report_output(Layer *layer,
        Output *output, OutputType output_type) {
    bool verbose = false;

    // Copy data over
    guint8* data = this->pixbufs[layer_indices[layer]]->get_pixels();

    float* output_count = output_count_map[layer->id];
    int max = 0;
    int min = -1;
    float avg = 0;
    int num_spiked = 0;

    if (iterations % integration_window == 0)
        for (int j = 0; j < layer->size; ++j)
            output_count[j] = 0;

    for (int j = 0; j < layer->size; ++j) {
        switch (output_type) {
            case FLOAT: output_count[j] += output[j].f; break;
            case BIT: if (output[j].i & (1 << 31)) ++output_count[j]; break;
            case INT: output_count[j] += output[j].i; break;
        }
        if (output_count[j] > 0) {
            if (output_count[j] > max) max = output_count[j];
            if ((min == -1) or (output_count[j] < min)) min = output_count[j];
            ++num_spiked;
        }
        avg += output_count[j];
    }
    avg /= layer->size;

    float std_dev = 0.0;
    for (int j = 0; j < layer->size; ++j)
        std_dev += pow((output_count[j] - avg), 2);
    std_dev = pow((std_dev / layer->size), 0.5);

    if (verbose and max != min)
        printf("%-10s -> %-10s: max %6d  min %6d  num_spiked %6d  "
               "avg %10.4f  std_dev %10.4f\n",
            layer->structure->name.c_str(),
            layer->name.c_str(),
            max, min, num_spiked, avg, std_dev / avg);

    for (int j = 0; j < layer->size; ++j) {
        if (output_count[j] == 0) {
            data[j*4 + 0] = 0.0;
            data[j*4 + 1] = 0.0;
            data[j*4 + 2] = 0.0;
        } else {
            float val = (max == min) ? 0.5 : ((output_count[j]-min) / (max-min));
            if (not linear) val *= val; // nonlinear heatmap
            data[j*4 + 0] = 255 * val;
            data[j*4 + 1] = 0.0;
            data[j*4 + 2] = 255.0 * (1.0 - val);
        }
    }

    if (layer->rows == 1)
        for (int j = 1; j < 50; ++j) {
            int index = j*layer->size * 4;
            for (int k = 0; k < layer->size; ++k) {
                data[index + (k*4 + 0)] = data[k*4 + 0];
                data[index + (k*4 + 1)] = data[k*4 + 1];
                data[index + (k*4 + 2)] = data[k*4 + 2];
            }
        }
}

void HeatmapWindowImpl::cycle() {
    ++iterations;
}
