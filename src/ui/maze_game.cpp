#include <climits>

#include "maze_game.h"
#include "maze_game_window.h"
#include "gui.h"
#include "model/layer.h"
#include "io/environment.h"

int MazeGame::instance_id = -1;

MazeGame *MazeGame::get_instance(bool init) {
    int id = MazeGame::instance_id;
    if ((id == -1 or id >= Frontend::instances.size())) {
        if (init) {
            new MazeGame();
            MazeGame::instance_id = Frontend::instances.size()-1;
        } else {
            MazeGame::instance_id = -1;
            return nullptr;
        }
    }
    return (MazeGame*)Frontend::instances[MazeGame::instance_id];
}

MazeGame::MazeGame() {
    this->maze_window = new MazeGameWindow();
    Frontend::set_window(this->maze_window);
}

MazeGame::~MazeGame() { }

bool MazeGame::add_input_layer(Layer *layer, std::string params) {
    // Check for duplicates
    try {
        auto info = layer_map.at(layer);
        return false;
    } catch (...) { }

    LayerInfo* info = new LayerInfo(layer);
    layer_map[layer] = info;
    info->set_input();
    return true;
}

bool MazeGame::add_output_layer(Layer *layer, std::string params) {
    // Check for duplicates
    try {
        auto info = layer_map.at(layer);
        return false;
    } catch (...) { }

    LayerInfo* info = new LayerInfo(layer);
    layer_map[layer] = info;
    info->set_output();
    return true;
}

void MazeGame::update(Environment *environment) {
    // Copy data over
    for (int i = 0; i < maze_window->layers.size(); ++i) {
        LayerInfo *info = maze_window->layers[i];
        if (info->get_output()) {
            Buffer *buffer = environment->buffer;
            Output *output = buffer->get_output(info->layer);
            OutputType output_type = environment->get_output_type(info->layer);
        }
    }

    // Signal GUI to update
    this->gui->dispatcher.emit();
}
