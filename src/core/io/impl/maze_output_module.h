#ifndef maze_output_module_h
#define maze_output_module_h

#include "io/module.h"
#include "maze_game.h"

class MazeOutputModule : public Module {
    public:
        MazeOutputModule(LayerList layers, ModuleConfig *config);

        void report_output(Buffer *buffer);

    private:
        MazeGame* maze_game;
        float threshold;
        int wait;

    MODULE_MEMBERS
};

#endif
