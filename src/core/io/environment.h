#ifndef environment_h
#define environment_h

#include "model/model.h"
#include "io/buffer.h"
#include "io/module/module.h"
#include "visualizer.h"

class Environment {
    public:
        Environment(Model *model, Buffer *buffer);
        virtual ~Environment();

        void step_input();
        void step_output();
        void ui_launch();
        void ui_update();

    private:
        Buffer *buffer;
        Visualizer *visualizer;
        ModuleList input_modules;
        ModuleList output_modules;
};

#endif
