#ifndef maze_game_window_h
#define maze_game_window_h

#include "network/layer.h"
#include "util/constants.h"

class MazeGameWindow {
    public:
        static MazeGameWindow* build();
        virtual void add_layer(Layer *layer, IOTypeMask io_type) = 0;

        virtual void set_cell_clear(int row, int col) = 0;
        virtual void set_cell_player(int row, int col) = 0;
        virtual void set_cell_goal(int row, int col) = 0;
};

#endif
