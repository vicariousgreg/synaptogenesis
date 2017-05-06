#include <climits>

#include "maze_game.h"
#include "maze_game_window.h"
#include "gui.h"
#include "model/layer.h"
#include "io/environment.h"
#include "util/tools.h"

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
    this->input_strength = 5.0;
    this->set_board_dim(3);
    this->maze_window = new MazeGameWindow(this);
    Frontend::set_window(this->maze_window);
}

MazeGame::~MazeGame() {
    input_data["player"].free();
    input_data["goal"].free();
    input_data["reward"].free();
    input_data["modulate"].free();
    input_data["somatosensory"].free();
    input_data["wall_left"].free();
    input_data["wall_right"].free();
    input_data["wall_up"].free();
    input_data["wall_down"].free();
}

void MazeGame::set_board_dim(int size) {
    this->board_dim = size;

    for (auto pair : input_data) pair.second.free();
    input_data.clear();

    input_data["player"] = Pointer<float>(board_dim*board_dim);
    input_data["goal"] = Pointer<float>(board_dim*board_dim);
    input_data["reward"] = Pointer<float>(1);
    input_data["modulate"] = Pointer<float>(1);
    input_data["somatosensory"] = Pointer<float>(4);
    input_data["wall_left"] = Pointer<float>(board_dim*board_dim);
    input_data["wall_right"] = Pointer<float>(board_dim*board_dim);
    input_data["wall_up"] = Pointer<float>(board_dim*board_dim);
    input_data["wall_down"] = Pointer<float>(board_dim*board_dim);
}

void MazeGame::init() {
    this->maze_window->init();
    this->ui_dirty = true;

    dirty["player"] = true;
    dirty["goal"] = true;
    dirty["reward"] = true;
    dirty["modulate"] = true;
    dirty["somatosensory"] = true;
    dirty["wall_left"] = true;
    dirty["wall_right"] = true;
    dirty["wall_up"] = true;
    dirty["wall_down"] = true;

    // Set up player and goal
    goal_row = player_row = fRand(0.0, board_dim);
    goal_col = player_col = fRand(0.0, board_dim);
    input_data["player"][player_row * board_dim + player_col] = input_strength;

    reset_goal();

    add_player();
    maze_window->set_cell_goal(goal_row, goal_col);

    // TODO: set up walls
}

bool MazeGame::is_dirty(std::string params) {
    try {
        return dirty.at(params);
    } catch (...) {
        ErrorManager::get_instance()->log_error(
            "Unrecognized params in maze_game " + params);
    }
}

Pointer<float> MazeGame::get_input(std::string params) {
    try {
        if (params == "reward") {
            float reward = input_data[params][0] * 0.95;
            input_data[params][0] = reward;
            dirty[params] = reward > 0.1;
        } else if (params == "modulate") {
            float modulate = input_data[params][0] * 0.95;
            input_data[params][0] = modulate;
            dirty[params] = modulate > 0.1;
        } else if (params == "somatosensory") {
            dirty[params] = false;

            for (int i = 0 ; i < 4 ; ++i) {
                float val = input_data[params][i];
                input_data[params][i] = 0.95 * val;
                if (val > 0.1) dirty[params] = true;
            }
        } else {
            dirty[params] = false;
        }
        return input_data.at(params);
    } catch (...) {
        ErrorManager::get_instance()->log_error(
            "Unrecognized params in maze_game " + params);
    }
}

bool MazeGame::add_input_layer(Layer *layer, std::string params) {
    // Check for duplicates
    try {
        input_data.at(params);
        auto info = layer_map.at(layer);
        return false;
    } catch (...) { }

    // Check layer size
    // Should be the size of the board
    if (layer->size != input_data[params].get_size()) return false;

    LayerInfo* info = new LayerInfo(layer);
    this->add_layer(layer, info);
    info->set_input();
    return true;
}

bool MazeGame::add_output_layer(Layer *layer, std::string params) {
    // Check for duplicates
    try {
        auto info = layer_map.at(layer);
        return false;
    } catch (...) { }

    // Check layer size
    // Should be 4 for output layer
    if (layer->size != 4) return false;

    LayerInfo* info = new LayerInfo(layer);
    this->add_layer(layer, info);
    info->set_output();
    return true;
}

void MazeGame::reset_goal() {
    // Remove goal
    input_data["goal"][goal_row * board_dim + goal_col] = 0.0;

    // Place new goal
    while (goal_row == player_row) goal_row = fRand(0.0, board_dim);
    while (goal_col == player_col) goal_col = fRand(0.0, board_dim);
    input_data["goal"][goal_row * board_dim + goal_col] = input_strength;

    time_to_reward = 0;
    moves_to_reward = 0;
    successful_moves = 0;
    min_moves = abs(goal_row - player_row) + abs(goal_col - player_col);

    ui_dirty = true;
    dirty["goal"] = true;
    dirty["reward"] = true;
    maze_window->set_cell_goal(goal_row, goal_col);
}

void MazeGame::administer_reward() {
    input_data["reward"][0] = input_strength;
    printf("Good job!  %6d iterations, %6d / %6d moves successful (%8.4f%%)"
           "     Minimum moves: %2d (%8.4f%% efficiency)\n",
        time_to_reward, successful_moves, moves_to_reward,
        (100.0 * successful_moves / moves_to_reward),
        min_moves,
        (100.0 * min_moves / moves_to_reward));
}

void MazeGame::remove_player() {
    input_data["player"][player_row * board_dim + player_col] = 0.0;
    maze_window->set_cell_clear(player_row, player_col);
    dirty["player"] = true;
    ui_dirty = true;
}

void MazeGame::add_player() {
    input_data["player"][player_row * board_dim + player_col] = input_strength;
    maze_window->set_cell_player(player_row, player_col);
    dirty["player"] = true;
    ui_dirty = true;

    // The player has reached the goal, so move it
    if (player_row == goal_row and player_col == goal_col) {
        administer_reward();
        reset_goal();
    }
}

bool MazeGame::move_up() {
    ++moves_to_reward;
    ++total_moves;
    if (player_row > 0) {
        ++successful_moves;
        ++total_successful_moves;
        remove_player();
        --player_row;
        add_player();
        input_data["somatosensory"][0] = input_strength;
        dirty["somatosensory"] = true;
        input_data["modulate"][0] = input_strength;
        dirty["modulate"] = true;
        return true;
    }
    return false;
}

bool MazeGame::move_down() {
    ++moves_to_reward;
    ++total_moves;
    if (player_row < board_dim-1) {
        ++successful_moves;
        ++total_successful_moves;
        remove_player();
        ++player_row;
        add_player();
        input_data["somatosensory"][1] = input_strength;
        dirty["somatosensory"] = true;
        input_data["modulate"][0] = input_strength;
        dirty["modulate"] = true;
        return true;
    }
    return false;
}

bool MazeGame::move_left() {
    ++moves_to_reward;
    ++total_moves;
    if (player_col > 0) {
        ++successful_moves;
        ++total_successful_moves;
        remove_player();
        --player_col;
        add_player();
        input_data["somatosensory"][2] = input_strength;
        dirty["somatosensory"] = true;
        input_data["modulate"][0] = input_strength;
        dirty["modulate"] = true;
        return true;
    }
}

bool MazeGame::move_right() {
    ++moves_to_reward;
    ++total_moves;
    if (player_col < board_dim-1) {
        ++successful_moves;
        ++total_successful_moves;
        remove_player();
        ++player_col;
        add_player();
        input_data["somatosensory"][3] = input_strength;
        dirty["somatosensory"] = true;
        input_data["modulate"][0] = input_strength;
        dirty["modulate"] = true;
        return true;
    }
    return false;
}

void MazeGame::update(Environment *environment) {
    ++iterations;
    ++time_to_reward;

    if (iterations % 1000 == 0) {
        printf("    %6d / %6d total moves successful (%8.4f%%)\n",
            total_successful_moves,
            total_moves,
            100.0 * total_successful_moves / total_moves);
        total_moves = 0;
        total_successful_moves = 0;
    }
    if (ui_dirty) {
        ui_dirty = false;

        // Signal GUI to update
        this->gui->dispatcher.emit();
    }
}
