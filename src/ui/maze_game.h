#ifndef maze_game_h
#define maze_game_h

#include <map>
#include <string>

#include "frontend.h"

class MazeGameWindow;
class Layer;
class Environment;

class MazeGame : public Frontend {
    public:
        static MazeGame *get_instance(bool init);

        virtual ~MazeGame();

        bool add_input_layer(Layer *layer, std::string params);
        bool add_output_layer(Layer *layer, std::string params);
        void update(Environment *environment);

    private:
        static int instance_id;
        MazeGame();

        MazeGameWindow *maze_window;
};

#endif
