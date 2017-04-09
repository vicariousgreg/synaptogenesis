#ifndef gui_h
#define gui_h

#include <gtkmm.h>

#include "layer_info.h"

class GUI {
    public:
        GUI();
        virtual ~GUI();

        void add_layer(LayerInfo *layer_info);
        void launch();
        void update();

        Glib::Dispatcher dispatcher;
        std::vector<LayerInfo*> layers;
        std::vector<Gtk::Image*> images;
        std::vector<Glib::RefPtr<Gdk::Pixbuf> > pixbufs;

    private:
        char** argv;
        Glib::RefPtr<Gtk::Application> app;
        Gtk::Window *window;
        Gtk::Grid *grid;
};

#endif
