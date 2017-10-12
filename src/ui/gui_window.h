#ifndef gui_window_h
#define gui_window_h

#include <gtkmm.h>

#include "network/layer.h"
#include "util/constants.h"

/* Wrapper class that ensures GUI is initialized */
class GuiObject { public: GuiObject(); };

class GuiWindow : public GuiObject, public Gtk::Window {
    public:
        GuiWindow();
        virtual void add_layer(Layer *layer, IOTypeMask io_type) = 0;
        virtual void update() = 0;
        virtual bool on_delete_event(GdkEventAny* any_event);

        std::vector<Layer*> layers;
};

#endif
