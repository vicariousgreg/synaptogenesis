#ifndef environment_h
#define environment_h

#include "model/structure.h"

class Visualizer;
class State;
class HostBuffer;
class Module;
typedef std::vector<Module*> ModuleList;

class Environment {
    public:
        Environment(State *state);
        virtual ~Environment();

        void step_input();
        void step_output();
        void ui_launch();
        void ui_update();

        OutputType get_output_type(Layer *layer);

        HostBuffer* const buffer;
        State* const state;

    private:
        Visualizer *visualizer;
        ModuleList input_modules;
        ModuleList output_modules;
};

#endif
