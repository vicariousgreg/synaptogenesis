#include <climits>

#include "dsst.h"
#include "dsst_window.h"
#include "gui.h"
#include "model/layer.h"
#include "io/environment.h"
#include "util/tools.h"

std::string DSST::name = "dsst";

DSST *DSST::get_instance(bool init) {
    auto instance = (DSST*)Frontend::get_instance(DSST::name);
    if (instance != nullptr)
        return instance;
    else if (init)
        return new DSST();
    else
        return nullptr;
}

DSST::DSST() {
    this->dsst_window = new DSSTWindow(this);
    Frontend::set_window(this->dsst_window);
    input_data = Pointer<float>(dsst_window->get_input_size());
    dsst_window->update_input(input_data);
}

DSST::~DSST() {
    input_data.free();
}

void DSST::init() {
    this->dsst_window->init();
    this->ui_dirty = true;
    this->input_dirty = true;
}

bool DSST::is_dirty(std::string params) {
    return input_dirty;
}

Pointer<float> DSST::get_input(std::string params) {
    input_dirty = false;
    return input_data;
}

bool DSST::add_input_layer(Layer *layer, std::string params) {
    // Check for duplicates
    try {
        auto info = layer_map.at(layer);
        return false;
    } catch (std::out_of_range) { }

    LayerInfo* info = new LayerInfo(layer);
    this->add_layer(layer, info);
    info->set_input();
    return true;
}

bool DSST::add_output_layer(Layer *layer, std::string params) {
}

void DSST::update(Environment *environment) {
    if (ui_dirty) {
        ui_dirty = false;

        // Signal GUI to update
        this->gui->dispatcher.emit();
    }
}

int DSST::get_input_rows() {
    return dsst_window->get_input_rows();
}

int DSST::get_input_columns() {
    return dsst_window->get_input_columns();
}

