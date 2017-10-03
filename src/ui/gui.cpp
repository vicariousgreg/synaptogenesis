#include <iostream>

#include "gui.h"
#include "gui_window.h"
#include "engine/engine.h"

GUI *GUI::instance = nullptr;

GUI *GUI::get_instance() {
    if (GUI::instance == nullptr)
        GUI::instance = new GUI();
    return GUI::instance;
}

GUI::GUI() : active(false), engine(nullptr) {
    update_dispatcher.connect(sigc::mem_fun(*this, &GUI::update));
    quit_dispatcher.connect(sigc::mem_fun(*this, &GUI::quit));
}

GUI::~GUI() {
    free(this->argv);
}

void GUI::add_window(GuiWindow *window) {
    active = true;
    this->windows.push_back(window);
}

void GUI::init(Engine *engine) {
    this->engine = engine;

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

void GUI::launch() {
    for (auto window : windows) window->show();
    if (windows.size() > 0)
        app->run(*windows[0]);
}

void GUI::signal_update() {
    if (active) update_dispatcher.emit();
}

void GUI::update() {
    for (auto window : windows) window->update();
}

void GUI::signal_quit() {
    if (active) quit_dispatcher.emit();
}

void GUI::quit() {
    for (auto window : windows) {
        window->close();
        delete window;
    }
    windows.clear();
    active = false;
}

void GUI::interrupt_engine() {
    if (engine != nullptr) engine->interrupt();
    engine = nullptr;
}
