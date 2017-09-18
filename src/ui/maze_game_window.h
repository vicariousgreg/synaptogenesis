#ifndef maze_game_window_h
#define maze_game_window_h

#include "gui_window.h"

class MazeGame;

class MazeGameWindow : public GuiWindow {
    public:
        MazeGameWindow(MazeGame *maze_game);
        virtual ~MazeGameWindow();

        void add_layer(LayerInfo *layer_info);
        void init();
        void update();

        void set_cell_clear(int row, int col);
        void set_cell_player(int row, int col);
        void set_cell_goal(int row, int col);

    private:
        void set_cell(int row, int col, int r, int g, int b);
        std::vector<Gtk::Image*> images;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> pixbufs;

        MazeGame* maze_game;
        Gtk::Grid *grid;
        int cell_size;
        int board_dim;
};

#endif
