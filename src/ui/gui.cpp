#include "gui.h"

#include <iostream>

GUI *GUI::instance = nullptr;

GUI *GUI::get_instance() {
    if (GUI::instance == nullptr)
        GUI::instance = new GUI();
    return GUI::instance;
}

void GUI::delete_instance() {
    if (GUI::instance != nullptr)
        delete GUI::instance;
    GUI::instance = nullptr;
}

GUI::GUI() {
    // Mock arguments
    int argc = 1;
    this->argv = (char**)malloc(sizeof(char*));
    this->argv[0] = " ";
    app =
        Gtk::Application::create(argc, this->argv,
                "org.gtkmm.examples.base");
    dispatcher.connect(sigc::mem_fun(*this, &GUI::update));
}


GUI::~GUI() {
    free(this->argv);
}

void GUI::add_window(GuiWindow *window) {
    this->windows.push_back(window);
}

void GUI::launch() {
    if (windows.size() > 0)
        app->run(*windows[0]);
}

void GUI::update() {
    for (auto window : windows) window->update();
}
