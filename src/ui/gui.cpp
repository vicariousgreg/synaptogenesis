#include <iostream>

#include "gui.h"
#include "gui_window.h"

GUI *GUI::instance = nullptr;

GUI *GUI::get_instance() {
    if (GUI::instance == nullptr)
        GUI::instance = new GUI();
    return GUI::instance;
}

GUI::GUI() : active(false), argv(nullptr) {
    update_dispatcher.connect(sigc::mem_fun(*this, &GUI::update));
    quit_dispatcher.connect(sigc::mem_fun(*this, &GUI::quit));
}

GUI::~GUI() {
    free(this->argv);
    GUI::instance = nullptr;
}

void GUI::add_window(GuiWindow *window) {
    this->windows.push_back(window);
}

void GUI::init() {
    if (not active) {
        active = true;

        // Mock arguments
        int argc = 1;
        if (this->argv == nullptr) {
            this->argv = (char**)malloc(sizeof(char*));
            this->argv[0] = " ";
        }
        app =
            Gtk::Application::create(argc, this->argv,
                    "org.gtkmm.examples.base");
    }
}

void GUI::launch() {
    if (active) {
        for (auto window : windows)
            window->show_now();

        if (windows.size() > 0)
            app->run(*windows[0]);
    }
}

void GUI::hide_windows() {
    for (auto window : windows) window->hide();
}

void GUI::signal_update() {
    if (active) update_dispatcher.emit();
}

void GUI::update() {
    if (active)
        for (auto window : windows)
            window->update();
}

void GUI::signal_quit() {
    if (active) quit_dispatcher.emit();
}

void GUI::quit() {
    if (active and windows.size() > 0) {
        active = false;
        for (auto window : windows) {
            window->close();
            delete window;
        }
        windows.clear();
    }
}
