#ifndef environment_h
#define environment_h

#include "model/structure.h"
#include "io/module/module.h"
#include "io/buffer.h"

class Visualizer;
class State;

class Environment {
    public:
        Environment(State *state, bool suppress_output=false);
        virtual ~Environment();

        void step_input();
        void step_output();
        void ui_init();
        void ui_launch();
        void ui_update();

        OutputType get_output_type(Layer *layer);

        Buffer* const buffer;
        State* const state;

    private:
        Visualizer *visualizer;
        ModuleList input_modules;
        ModuleList expected_modules;
        ModuleList output_modules;
};

#endif
