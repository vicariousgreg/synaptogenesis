#ifndef random_input_module_h
#define random_input_module_h

#include "io/module.h"

class RandomInputModule : public Module {
    public:
        RandomInputModule(Layer *layer, ModuleConfig *config);
        virtual ~RandomInputModule();

        void feed_input(Buffer *buffer);
        void cycle();

    private:
        int timesteps;
        int shuffle_rate;
        float max_value;
        float fraction;
        float *random_values;
        bool verbose;
        bool clear;
        bool uniform;

    MODULE_MEMBERS
};

#endif
