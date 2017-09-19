#ifndef visualizer_window_impl_h
#define visualizer_window_impl_h

#include "visualizer_window.h"
#include "gui_window.h"

class VisualizerWindowImpl : public VisualizerWindow, public GuiWindow {
    public:
        VisualizerWindowImpl();
        virtual ~VisualizerWindowImpl();

        void update();
        void add_layer(Layer *layer, IOTypeMask io_type);
        void feed_input(Layer *layer, float *input);
        void report_output(Layer *layer,
            Output *output, OutputType output_type);

    protected:
        Gtk::Grid *grid;
        std::vector<Gtk::Image*> images;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> pixbufs;
        std::map<Layer*, int> layer_indices;
};

class HeatmapWindowImpl : public VisualizerWindowImpl {
    public:
        HeatmapWindowImpl();
        virtual ~HeatmapWindowImpl();

        void add_layer(Layer *layer, IOTypeMask io_type);
        void feed_input(Layer *layer, float *input);
        void report_output(Layer *layer,
            Output *output, OutputType output_type);

    protected:
        int iterations;
        std::map<int, float*> spike_count_map;
};

#endif
