#ifndef maze_game_window_h
#define maze_game_window_h

#include "gui_window.h"

class MazeGameWindow : public GuiWindow {
    public:
        MazeGameWindow();
        virtual ~MazeGameWindow();

        void add_layer(LayerInfo *layer_info);
        void update();
};

#endif
