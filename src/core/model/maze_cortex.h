#ifndef maze_cortex_h
#define maze_cortex_h

#include <vector>
#include <string>

#include "model/model.h"
#include "model/structure.h"

class MazeCortex : public Structure {
    public:
        MazeCortex(Model *model, int board_dim, int cell_size);

    protected:
        int board_dim;
        int cell_size;
        int cortex_size;

        void add_cortical_layer(std::string name, int size_fraction=1);
        void connect_one_way(std::string name1, std::string name2,
            int spread, float fraction, int delay=0, int stride=1);
        void add_input_grid(std::string layer, std::string input_name,
            ModuleConfig *config);
        void add_input_random(std::string layer, std::string input_name,
            ModuleConfig *config);
};

#endif