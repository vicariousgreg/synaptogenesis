#include <algorithm>

#include "io/impl/maze_module.h"
#include "util/error_manager.h"

REGISTER_MODULE(MazeModule, "maze_game");

MazeModule::MazeModule(LayerList layers, ModuleConfig *config)
        : Module(layers), threshold(1), wait(0) {
    this->input_strength = 5.0;
    this->window = MazeGameWindow::build();

    this->ui_dirty = true;

    input_data["player"] = Pointer<float>(board_dim*board_dim);
    input_data["goal"] = Pointer<float>(board_dim*board_dim);
    input_data["reward"] = Pointer<float>(1);
    input_data["modulate"] = Pointer<float>(1);
    input_data["somatosensory"] = Pointer<float>(4);
    input_data["wall_left"] = Pointer<float>(board_dim*board_dim);
    input_data["wall_right"] = Pointer<float>(board_dim*board_dim);
    input_data["wall_up"] = Pointer<float>(board_dim*board_dim);
    input_data["wall_down"] = Pointer<float>(board_dim*board_dim);
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
    window->set_cell_goal(goal_row, goal_col);

    // TODO: set up walls

    for (auto layer : layers) {
        auto param =
            config->get_layer(layer)->get_property("params", "");
        params[layer] = param;
        if (param == "")
            ErrorManager::get_instance()->log_error(
                "Unspecified MazeModule layer parameter!");

        if (param == "input") {
            // Check layer size
            // Should be the size of the board
            if (layer->size != input_data[param].get_size())
                ErrorManager::get_instance()->log_error(
                    "Mismatched MazeModule input layer size!");

            set_io_type(layer, INPUT);
            window->add_layer(layer, INPUT);
        } else if (param == "output") {
            // Check layer size
            // Should be 4 for output layer
            if (layer->size != 4)
                ErrorManager::get_instance()->log_error(
                    "Mismatched MazeModule output layer size!");

            set_io_type(layer, OUTPUT);
            window->add_layer(layer, OUTPUT);
        } else {
            ErrorManager::get_instance()->log_error(
                "Unrecognized layer type: " + param
                + " in VisualizerModule!");
        }
        set_io_type(INPUT);
    }
}

MazeModule::~MazeModule() {
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

void MazeModule::feed_input(Buffer *buffer) {
    for (auto layer : layers) {
        if (is_dirty(params[layer])) {
            Pointer<float> input = get_input(params[layer]);
            buffer->set_input(layer, input);
        }
    }
}

static float convert_spikes(unsigned int spikes) {
    int count = 0;
    for (int i = 0; i < 31; ++i)
        if (spikes & (1 << (32 - i))) ++count;
    return count;
}

void MazeModule::report_output(Buffer *buffer) {
    if (wait > 0) {
        --wait;
        return;
    }

    for (auto layer : layers) {
        Output* output = buffer->get_output(layer);

        float up, down, left, right;
        switch(output_types[layer]) {
            case BIT:
                up = convert_spikes(output[0].i);
                down = convert_spikes(output[1].i);
                left = convert_spikes(output[2].i);
                right = convert_spikes(output[3].i);
                break;
            case FLOAT:
                up = output[0].f;
                down = output[1].f;
                left = output[2].f;
                right = output[3].f;
                break;
        }

        float max = std::max(up, std::max(down, std::max(left, right)));
        int num_max = (up == max) + (down == max) + (left == max) + (right == max);
        if (max >= threshold and num_max == 1) {
            bool success = false;
            if (up == max) success = move_up();
            else if (down == max) success = move_down();
            else if (left == max) success = move_left();
            else if (right == max) success = move_right();
            //if (success) wait = 20;
            wait = 20;
        }
    }
}

bool MazeModule::is_dirty(std::string params) {
    try {
        return dirty.at(params);
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unrecognized params in maze_game " + params);
    }
}

Pointer<float> MazeModule::get_input(std::string params) {
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
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unrecognized params in maze_game " + params);
    }
}

void MazeModule::reset_goal() {
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
    window->set_cell_goal(goal_row, goal_col);
}

void MazeModule::administer_reward() {
    input_data["reward"][0] = input_strength;
    printf("Good job!  %6d iterations, %6d / %6d moves successful (%8.4f%%)"
           "     Minimum moves: %2d (%8.4f%% efficiency)\n",
        time_to_reward, successful_moves, moves_to_reward,
        (100.0 * successful_moves / moves_to_reward),
        min_moves,
        (100.0 * min_moves / moves_to_reward));
}

void MazeModule::remove_player() {
    input_data["player"][player_row * board_dim + player_col] = 0.0;
    window->set_cell_clear(player_row, player_col);
    dirty["player"] = true;
    ui_dirty = true;
}

void MazeModule::add_player() {
    input_data["player"][player_row * board_dim + player_col] = input_strength;
    window->set_cell_player(player_row, player_col);
    dirty["player"] = true;
    ui_dirty = true;

    // The player has reached the goal, so move it
    if (player_row == goal_row and player_col == goal_col) {
        administer_reward();
        reset_goal();
    }
}

bool MazeModule::move_up() {
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

bool MazeModule::move_down() {
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

bool MazeModule::move_left() {
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

bool MazeModule::move_right() {
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

void MazeModule::cycle() {
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
    }
}