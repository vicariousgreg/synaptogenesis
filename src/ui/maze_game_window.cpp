#include "maze_game_window.h"
#include "impl/maze_game_window_impl.h"

MazeGameWindow* MazeGameWindow::build() {
    return new MazeGameWindowImpl();
}
