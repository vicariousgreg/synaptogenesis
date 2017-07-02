#include "io/module/maze_input_module.h"
#include "util/error_manager.h"

REGISTER_MODULE(MazeInputModule, "maze_input", INPUT);

MazeInputModule::MazeInputModule(Layer *layer, ModuleConfig *config)
        : Module(layer), params(config->get_property("params")) {
    maze_game = MazeGame::get_instance(true);
    if (not maze_game->add_input_layer(layer, params))
        ErrorManager::get_instance()->log_error(
            "Failed to add layer to Maze Game!");
}

void MazeInputModule::feed_input(Buffer *buffer) {
    if (maze_game->is_dirty(params)) {
        Pointer<float> input = maze_game->get_input(params);
        buffer->set_input(this->layer, input);
    }
}
