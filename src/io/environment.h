#ifndef environment_h
#define environment_h

#include "model/model.h"
#include "io/buffer.h"

class Environment {
    public:
        Environment(Model *model) : model(model) { }

        void step_input(Buffer *buffer);
        void step_output(Buffer *buffer);

    private:
        Model *model;
};

#endif
