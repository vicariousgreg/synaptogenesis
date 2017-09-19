#ifndef maze_game_window_impl_h
#define maze_game_window_impl_h

#include "maze_game_window.h"
#include "gui_window.h"

class MazeGameWindowImpl : public MazeGameWindow, public GuiWindow {
    public:
        MazeGameWindowImpl();
        virtual ~MazeGameWindowImpl();

        void update();
        void add_layer(Layer *layer, IOTypeMask io_type);

        void set_cell_clear(int row, int col);
        void set_cell_player(int row, int col);
        void set_cell_goal(int row, int col);

    protected:
        void set_cell(int row, int col, int r, int g, int b);

        std::vector<Gtk::Image*> images;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> pixbufs;

        Gtk::Grid *grid;
        int cell_size;
        int board_dim;
};

#endif
