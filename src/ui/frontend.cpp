#include "frontend.h"
#include "gui.h"
#include "gui_window.h"
#include "io/environment.h"

std::vector<Frontend*> Frontend::instances;

Frontend::Frontend() : gui_window(nullptr) {
    this->gui = GUI::get_instance();
    Frontend::instances.push_back(this);
}

void Frontend::set_window(GuiWindow *gui_window) {
    this->gui_window = gui_window;
    this->gui->add_window(gui_window);
}

Frontend::~Frontend() {
    if (gui_window != nullptr) delete gui_window;
    for (auto pair : layer_map) delete pair.second;
}

const std::vector<Frontend*> Frontend::get_instances() {
    return Frontend::instances;
}

void Frontend::init_all() {
    for (auto f : Frontend::instances)
        if (f->gui_window != nullptr)
            for (auto pair : f->layer_map)
                f->gui_window->add_layer(pair.second);
}

void Frontend::launch_all() {
    if (Frontend::instances.size() > 0)
        GUI::get_instance()->launch();
}

void Frontend::update_all(Environment *environment) {
    for (auto f : Frontend::instances)
        f->update(environment);
}

void Frontend::cleanup() {
    for (auto f : Frontend::instances) delete f;
    GUI::delete_instance();
}
