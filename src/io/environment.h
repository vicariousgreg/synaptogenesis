#ifndef environment_h
#define environment_h

#include "model/model.h"
#include "io/buffer.h"

class Environment {
    public:
        Environment(Model *model, Buffer *buffer);
        virtual ~Environment();

        void step_input();
        void step_output();

        void ui_init();
        void ui_update();

    private:
        Buffer *buffer;
        std::vector<Module*> input_modules;
        std::vector<Module*> output_modules;

        const char *fifo_name = "/tmp/pcnn_fifo";
        int fifo_fd;
};

#endif
