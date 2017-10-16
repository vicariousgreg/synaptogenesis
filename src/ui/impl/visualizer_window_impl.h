#ifndef visualizer_window_impl_h
#define visualizer_window_impl_h

#include "visualizer_window.h"
#include "gui_window.h"

class VisualizerWindowImpl : public VisualizerWindow, public GuiWindow {
    public:
        VisualizerWindowImpl();
        virtual ~VisualizerWindowImpl();

        virtual void update();
        virtual void add_layer(Layer *layer, IOTypeMask io_type);
        virtual void feed_input(Layer *layer, float *input);
        virtual void report_output(Layer *layer,
            Output *output, OutputType output_type);

    protected:
        Gtk::Grid *grid;
        std::vector<Gtk::Image*> images;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> pixbufs;
        std::map<Layer*, int> layer_indices;
};

class HeatmapWindowImpl : public VisualizerWindowImpl {
    public:
        HeatmapWindowImpl(int rate, bool linear);
        virtual ~HeatmapWindowImpl();

        virtual void add_layer(Layer *layer, IOTypeMask io_type);
        virtual void feed_input(Layer *layer, float *input);
        virtual void report_output(Layer *layer,
            Output *output, OutputType output_type);
        void cycle();

    protected:
        int iterations;
        int rate;
        bool linear;
        std::map<int, float*> output_count_map;
};

#endif
