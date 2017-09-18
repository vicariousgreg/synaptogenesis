#ifndef gui_h
#define gui_h

#include <vector>
#include <gtkmm.h>

#include "layer_info.h"
#include "gui_window.h"

class GUI {
    public:
        static GUI *get_instance();
        static void delete_instance();

        GUI();
        virtual ~GUI();

        void add_window(GuiWindow *window);
        void launch();
        void quit();
        void update();

        Glib::Dispatcher dispatcher;

    private:
        static GUI *instance;

        char** argv;
        Glib::RefPtr<Gtk::Application> app;
        std::vector<GuiWindow*> windows;
};

#endif
