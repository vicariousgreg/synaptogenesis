#ifndef gui_h
#define app_h

#include <gtkmm.h>

class GUI {
    public:
        GUI();
        virtual ~GUI();

        void launch();

    private:
        Glib::RefPtr<Gtk::Application> app;
        Gtk::Window *window;
};

#endif
