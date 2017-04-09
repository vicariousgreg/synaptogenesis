#ifndef maze_output_module_h
#define maze_output_module_h

#include "io/module/module.h"
#include "util/error_manager.h"
#include "maze_game.h"

class MazeOutputModule : public Module {
    public:
        MazeOutputModule(Layer *layer, std::string params)
            : Module(layer) {
            if (not MazeGame::get_instance(true)
                    ->add_output_layer(layer, params))
                ErrorManager::get_instance()->log_error(
                    "Failed to add layer to Maze Game!");
        }

        virtual IOTypeMask get_type() { return OUTPUT; }
};

#endif
