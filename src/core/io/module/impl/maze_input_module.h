#ifndef maze_input_module_h
#define maze_input_module_h

#include "io/module/module.h"
#include "maze_game.h"

class MazeInputModule : public Module {
    public:
        MazeInputModule(Layer *layer, ModuleConfig *config);

        void feed_input(Buffer *buffer);
        virtual IOTypeMask get_type() { return INPUT; }

    private:
        MazeGame* maze_game;
        std::string params;

    MODULE_MEMBERS
};

#endif
