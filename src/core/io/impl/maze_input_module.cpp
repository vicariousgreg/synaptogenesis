#include "io/impl/maze_input_module.h"
#include "util/error_manager.h"

REGISTER_MODULE(MazeInputModule, "maze_input", INPUT);

MazeInputModule::MazeInputModule(LayerList layers, ModuleConfig *config)
        : Module(layers) {
    enforce_equal_layer_sizes("maze_input");

    maze_game = MazeGame::get_instance(true);
    for (auto layer : layers) {
        params[layer] = config->get_property("params");
        if (not maze_game->add_input_layer(layer, params[layer]))
            ErrorManager::get_instance()->log_error(
                "Failed to add layer to Maze Game!");
    }
}

void MazeInputModule::feed_input(Buffer *buffer) {
    for (auto layer : layers) {
        if (maze_game->is_dirty(params[layer])) {
            Pointer<float> input = maze_game->get_input(params[layer]);
            buffer->set_input(layer, input);
        }
    }
}
