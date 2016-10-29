#ifndef gui_h
#define gui_h

#include <gtkmm.h>

#include "layer_info.h"

class GUI {
    public:
        GUI();
        virtual ~GUI();

        void add_layer(LayerInfo layer_info);
        void launch();
        void update();

        bool done;
        sigc::signal<void> signal_update;
        std::vector<LayerInfo> layers;
        std::vector<Gtk::Image*> images;
        std::vector<Glib::RefPtr<Gdk::Pixbuf> > pixbufs;

    private:
        Glib::RefPtr<Gtk::Application> app;
        Gtk::Window *window;
};

#endif
