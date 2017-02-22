#ifndef environment_h
#define environment_h

#include "model/structure.h"
#include "io/buffer.h"
#include "io/module/module.h"

class Visualizer;
class State;

class Environment {
    public:
        Environment(State *state);
        virtual ~Environment();

        void step_input();
        void step_output();
        void ui_launch();
        void ui_update();

        Buffer *get_buffer(Structure *structure) {
            return buffers.at(structure);
        }

    private:
        std::map<Structure*, Buffer*> buffers;
        Visualizer *visualizer;
        ModuleList input_modules;
        ModuleList output_modules;
};

#endif
