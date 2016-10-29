#include "gui.h"

GUI::GUI() {
    int argc = 0;
    char **argv = NULL;
    app =
        Gtk::Application::create(argc, argv,
                "org.gtkmm.examples.base");

    window = new Gtk::Window();
    window->set_default_size(200, 200);
}


GUI::~GUI() {
    delete this->window;
}

void GUI::launch() {
    app->run(*window);
}
