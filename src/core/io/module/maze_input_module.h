#ifndef maze_input_module_h
#define maze_input_module_h

#include "io/module/module.h"
#include "util/error_manager.h"
#include "maze_game.h"

class MazeInputModule : public Module {
    public:
        MazeInputModule(Layer *layer, std::string params)
            : Module(layer) {
            if (not MazeGame::get_instance(true)
                    ->add_input_layer(layer, params))
                ErrorManager::get_instance()->log_error(
                    "Failed to add layer to Maze Game!");
        }

        virtual IOTypeMask get_type() { return INPUT; }
};

#endif
