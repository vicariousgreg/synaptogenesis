#ifndef environment_h
#define environment_h

#include "model/model.h"
#include "model/structure.h"
#include "io/buffer.h"
#include "io/module/module.h"
#include "engine/engine.h"

class Visualizer;

class Environment {
    public:
        Environment(Model *model, Engine *engine);
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
