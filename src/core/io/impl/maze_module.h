#ifndef maze_module_h
#define maze_module_h

#include "io/module.h"
#include "maze_game_window.h"

class MazeModule : public Module {
    public:
        MazeModule(LayerList layers, ModuleConfig *config);
        virtual ~MazeModule();

        void feed_input(Buffer *buffer);
        void report_output(Buffer *buffer);

        void init();
        bool add_input_layer(Layer *layer, std::string params);
        bool add_output_layer(Layer *layer, std::string params);
        void cycle();

        Pointer<float> get_input(std::string params);
        bool is_dirty(std::string params);

        bool move_up();
        bool move_down();
        bool move_left();
        bool move_right();

        static const int board_dim = 3;

    private:
        float threshold;
        int wait;
        std::map<Layer*, std::string> params;

        void add_player();
        void remove_player();
        void reset_goal();
        void administer_reward();

        int iterations;
        int total_successful_moves;
        int total_moves;

        int time_to_reward;
        int moves_to_reward;
        int successful_moves;
        int min_moves;

        float input_strength;
        int player_row, player_col;
        int goal_row, goal_col;
        bool ui_dirty;
        std::map<std::string, bool > dirty;
        std::map<std::string, Pointer<float>> input_data;
        MazeGameWindow *window;

    MODULE_MEMBERS
};

#endif
