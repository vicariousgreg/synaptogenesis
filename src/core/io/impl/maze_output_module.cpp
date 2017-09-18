#include <algorithm>

#include "io/impl/maze_output_module.h"
#include "util/error_manager.h"

REGISTER_MODULE(MazeOutputModule, "maze_output", OUTPUT);

MazeOutputModule::MazeOutputModule(Layer *layer, ModuleConfig *config)
        : Module(layer), threshold(1), wait(0) {
    maze_game = MazeGame::get_instance(true);
    if (not maze_game->add_output_layer(layer, config->get_property("params")))
        ErrorManager::get_instance()->log_error(
            "Failed to add layer to Maze Game!");
}

static float convert_spikes(unsigned int spikes) {
    int count = 0;
    for (int i = 0; i < 31; ++i)
        if (spikes & (1 << (32 - i))) ++count;
    return count;
}

void MazeOutputModule::report_output(Buffer *buffer) {
    if (wait > 0) {
        --wait;
        return;
    }
    Output* output = buffer->get_output(this->layer);

    float up, down, left, right;
    switch(output_type) {
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
        if (up == max) success = maze_game->move_up();
        else if (down == max) success = maze_game->move_down();
        else if (left == max) success = maze_game->move_left();
        else if (right == max) success = maze_game->move_right();
        //if (success) wait = 20;
        wait = 20;
    }
}
