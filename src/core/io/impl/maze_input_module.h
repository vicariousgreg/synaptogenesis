#ifndef maze_input_module_h
#define maze_input_module_h

#include "io/module.h"
#include "maze_game.h"

class MazeInputModule : public Module {
    public:
        MazeInputModule(LayerList layers, ModuleConfig *config);

        void feed_input(Buffer *buffer);

    private:
        MazeGame* maze_game;
        std::map<Layer*, std::string> params;

    MODULE_MEMBERS
};

#endif
