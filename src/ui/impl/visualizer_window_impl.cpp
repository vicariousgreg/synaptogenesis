#include "visualizer_window.h"
#include "visualizer_window_impl.h"

#include "network/structure.h"

VisualizerWindowImpl::VisualizerWindowImpl() {
    grid = new Gtk::Grid();
    grid->set_row_spacing(1);
    grid->set_column_spacing(1);
    grid->override_background_color(Gdk::RGBA("DarkSlateGray"));
    this->add(*grid);
}

VisualizerWindowImpl::~VisualizerWindowImpl() {
    delete this->grid;
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

    for (int j = 0; j < layer->size; ++j) {
        float val = convert(output[j], output_type);
        data[j*4 + 0] = val;
        data[j*4 + 1] = val;
        data[j*4 + 2] = val;
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

HeatmapWindowImpl::HeatmapWindowImpl(int integration_window, bool linear)
    : iterations(1), integration_window(integration_window), linear(linear) { }

HeatmapWindowImpl::~HeatmapWindowImpl() {
    for (auto pair : output_count_map) free(pair.second);
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
