#ifndef gui_h
#define gui_h

#include <vector>
#include <gtkmm.h>

class GuiWindow;

class GUI {
    public:
        static GUI *get_instance();

        GUI();
        virtual ~GUI();

        bool is_active() { return active; }
        void add_window(GuiWindow *window);
        void launch();
        void quit();
        void signal_update();

        Glib::Dispatcher dispatcher;

    private:
        static GUI *instance;

        void init();
        void update();
        bool active;

        char** argv;
        Glib::RefPtr<Gtk::Application> app;
        std::vector<GuiWindow*> windows;
};

#endif
