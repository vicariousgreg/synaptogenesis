#ifndef gui_h
#define gui_h

#include <vector>
#include <gtkmm.h>

class GuiWindow;
class GuiController;

class GUI {
    public:
        static GUI *get_instance();
        virtual ~GUI();

        bool is_active() { return active; }
        void add_window(GuiWindow *window);
        void init();
        void launch();
        void signal_update();
        void signal_quit();
        void hide_windows();

        Glib::Dispatcher quit_dispatcher;
        Glib::Dispatcher update_dispatcher;

    private:
        friend GuiController;

        static GUI *instance;
        GUI();

        void update();
        void quit();
        bool active;

        char** argv;
        Glib::RefPtr<Gtk::Application> app;
        std::vector<GuiWindow*> windows;

        Gtk::Window *dummy_window;
};

#endif
