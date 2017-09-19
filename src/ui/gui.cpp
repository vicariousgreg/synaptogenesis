#include <iostream>

#include "gui.h"
#include "gui_window.h"

GUI *GUI::instance = nullptr;

GUI *GUI::get_instance() {
    if (GUI::instance == nullptr)
        GUI::instance = new GUI();
    return GUI::instance;
}

GUI::GUI() : active(false) {
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
    active = true;
    this->windows.push_back(window);
}

void GUI::launch() {
    for (auto window : windows) window->show();
    if (windows.size() > 0)
        app->run(*windows[0]);
}

void GUI::quit() {
    for (auto window : windows) window->close();
    if (windows.size() > 0)
        app->quit();
    for (auto window : windows) delete window;
    windows.clear();
    active = false;
}

void GUI::signal_update() {
    if (active) dispatcher.emit();
}

void GUI::update() {
    for (auto window : windows) window->update();
}
