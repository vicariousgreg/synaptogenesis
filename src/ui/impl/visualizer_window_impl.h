#ifndef visualizer_window_impl_h
#define visualizer_window_impl_h

#include "visualizer_window.h"
#include "gui_window.h"

class VisualizerWindowImpl : public VisualizerWindow, public GuiWindow {
    public:
        VisualizerWindowImpl(PropertyConfig *config);
        virtual ~VisualizerWindowImpl();

        virtual void update();
        virtual void add_layer(Layer *layer, IOTypeMask io_type);
        virtual void feed_input(Layer *layer, float *input);
        virtual void report_output(Layer *layer,
            Output *output, OutputType output_type);

    protected:
        bool colored;
        bool negative;
        bool decay;
        int color_window;
        int bump;
        float freq_r;
        float freq_g;
        float freq_b;

        Gtk::Grid *grid;
        std::vector<Gtk::Image*> images;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> pixbufs;
        std::map<Layer*, int> layer_indices;
        std::map<size_t, int*> hues;
};

class HeatmapWindowImpl : public VisualizerWindowImpl {
    public:
        HeatmapWindowImpl(PropertyConfig *config);
        virtual ~HeatmapWindowImpl();

        virtual void update();
        virtual void add_layer(Layer *layer, IOTypeMask io_type);
        virtual void feed_input(Layer *layer, float *input);
        virtual void report_output(Layer *layer,
            Output *output, OutputType output_type);
        void cycle();

    protected:
        int iterations;
        int integration_window;
        bool linear;
        bool stats;
        std::map<size_t, float*> output_count_map;
        Gtk::Label *label;
};

#endif
