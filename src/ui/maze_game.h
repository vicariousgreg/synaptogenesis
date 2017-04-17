#ifndef maze_game_h
#define maze_game_h

#include <map>
#include <string>

#include "frontend.h"
#include "util/pointer.h"

class MazeGameWindow;
class Layer;
class Environment;

class MazeGame : public Frontend {
    public:
        static MazeGame *get_instance(bool init);

        virtual ~MazeGame();

        void init();
        bool add_input_layer(Layer *layer, std::string params);
        bool add_output_layer(Layer *layer, std::string params);
        void update(Environment *environment);

        Pointer<float> get_input(std::string params);
        bool is_dirty(std::string params);

        void move_up();
        void move_down();
        void move_left();
        void move_right();

        int get_board_dim() { return board_dim; }

    private:
        friend class MazeGameWindow;

        static int instance_id;
        MazeGame();

        void add_player();
        void remove_player();

        int board_dim;
        int player_row, player_col;
        int goal_row, goal_col;
        bool ui_dirty;
        std::map<std::string, bool > dirty;
        std::map<std::string, Pointer<float> > input_data;
        MazeGameWindow *maze_window;
};

#endif
