#ifndef visualizer_window_h
#define visualizer_window_h

#include "gui_window.h"

class VisualizerWindow : public GuiWindow {
    public:
        VisualizerWindow();
        virtual ~VisualizerWindow();

        void add_layer(LayerInfo *layer_info);
        void update();

        std::vector<Gtk::Image*> images;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> pixbufs;

    private:
        Gtk::Grid *grid;
};

#endif
