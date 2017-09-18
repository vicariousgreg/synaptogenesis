#include "frontend.h"
#include "gui.h"
#include "gui_window.h"
#include "io/buffer.h"

std::vector<Frontend*> Frontend::instances;

Frontend::Frontend() : gui_window(nullptr) {
    this->gui = GUI::get_instance();
    Frontend::instances.push_back(this);
}

void Frontend::set_window(GuiWindow *gui_window) {
    this->gui_window = gui_window;
    this->gui->add_window(gui_window);
}

void Frontend::add_layer(Layer* layer, LayerInfo* info) {
    layer_list.push_back(layer);
    layer_map[layer] = info;
}

Frontend::~Frontend() {
    if (gui_window != nullptr) delete gui_window;
    for (auto pair : layer_map) delete pair.second;
}

Frontend* Frontend::get_instance(std::string name) {
    for (auto f : Frontend::instances)
        if (f->get_name() == name)
            return f;
    return nullptr;
}

void Frontend::init_all() {
    for (auto f : Frontend::instances) {
        f->init();
        if (f->gui_window != nullptr)
            for (auto layer : f->layer_list)
                f->gui_window->add_layer(f->layer_map[layer]);
    }
}

void Frontend::launch_all() {
    if (Frontend::instances.size() > 0)
        GUI::get_instance()->launch();
}

void Frontend::update_all(Buffer *buffer) {
    for (auto f : Frontend::instances)
        f->update(buffer);
}

void Frontend::cleanup() {
    for (auto f : Frontend::instances) delete f;
    Frontend::instances.clear();
    GUI::delete_instance();
}
