#ifndef maze_output_module_h
#define maze_output_module_h

#include "io/module.h"
#include "maze_game.h"

class MazeOutputModule : public Module {
    public:
        MazeOutputModule(Layer *layer, ModuleConfig *config);

        void report_output(Buffer *buffer, OutputType output_type);

    private:
        MazeGame* maze_game;
        float threshold;
        int wait;

    MODULE_MEMBERS
};

#endif
