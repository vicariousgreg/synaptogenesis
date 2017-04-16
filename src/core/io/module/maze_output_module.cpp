#include <algorithm>

#include "io/module/maze_output_module.h"
#include "util/error_manager.h"

MazeOutputModule::MazeOutputModule(Layer *layer, std::string params)
        : Module(layer) {
    maze_game = MazeGame::get_instance(true);
    if (not maze_game->add_output_layer(layer, params))
        ErrorManager::get_instance()->log_error(
            "Failed to add layer to Maze Game!");
}

static float convert_spikes(unsigned int spikes) {
    int count = 0;
    for (int i = 0; i < 8; ++i)
        if (spikes & (1 << (32 - i))) ++count;
    return count;
}

void MazeOutputModule::report_output(Buffer *buffer, OutputType output_type) {
    Output* output = buffer->get_output(this->layer);

    float up, down, left, right;
    switch(output_type) {
        case(BIT):
            up = convert_spikes(output[0].i);
            down = convert_spikes(output[1].i);
            left = convert_spikes(output[2].i);
            right = convert_spikes(output[3].i);
            break;
        case(FLOAT):
            up = output[0].f;
            down = output[1].f;
            left = output[2].f;
            right = output[3].f;
            break;
    }

    float max = std::max(up, std::max(down, std::max(left, right)));
    if (up == max) maze_game->move_up();
    else if (down == max) maze_game->move_down();
    else if (left == max) maze_game->move_left();
    else if (right == max) maze_game->move_right();
}
