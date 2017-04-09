#ifndef gui_window_h
#define gui_window_h

#include <gtkmm.h>

#include "layer_info.h"

class GuiWindow : public Gtk::Window {
    public:
        virtual void add_layer(LayerInfo *layer_info) = 0;
        virtual void update() = 0;

        std::vector<LayerInfo*> layers;
};

#endif
