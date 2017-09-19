#ifndef gui_window_h
#define gui_window_h

#include <gtkmm.h>

#include "network/layer.h"
#include "util/constants.h"

class GuiWindow : public Gtk::Window {
    public:
        GuiWindow();
        virtual void add_layer(Layer *layer, IOTypeMask io_type) = 0;
        virtual void update() = 0;

        std::vector<Layer*> layers;
};

#endif
