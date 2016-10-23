#ifndef environment_h
#define environment_h

#include "model/model.h"
#include "io/buffer.h"

class Environment {
    public:
        Environment(Model *model);
        virtual ~Environment() {
            for (int i = 0; i < this->input_modules.size(); ++i)
                delete this->input_modules[i];
            for (int i = 0; i < this->output_modules.size(); ++i)
                delete this->output_modules[i];
        }

        void step_input(Buffer *buffer);
        void step_output(Buffer *buffer);

    private:
        std::vector<Module*> input_modules;
        std::vector<Module*> output_modules;
};

#endif
