#ifndef maze_output_module_h
#define maze_output_module_h

#include "io/module/module.h"
#include "maze_game.h"

class MazeOutputModule : public Module {
    public:
        MazeOutputModule(Layer *layer, std::string params);

        void report_output(Buffer *buffer, OutputType output_type);
        virtual IOTypeMask get_type() { return OUTPUT; }

    private:
        MazeGame* maze_game;
        float threshold;
};

#endif
